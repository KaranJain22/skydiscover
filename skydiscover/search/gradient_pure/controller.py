"""
GradientPureController: fully gradient-based algorithm discovery.

No LLM in the inner loop. Maintains a population of PrimitiveGraphs
with WeightedChoice nodes. Each iteration:
1. Sample a graph from population
2. Gradient optimize all parameters (including architecture weights)
3. Discretize -> code
4. Evaluate
5. Update population scores

New structures are created by random perturbation of WeightedChoice weights.
"""

from __future__ import annotations

import copy
import logging
import os
import random
import time
import uuid
from typing import Optional

import torch

from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.differentiable.codegen import graph_to_solution
from skydiscover.search.differentiable.optimizer import (
    OptimizationConfig,
    PrimitiveOptimizer,
)
from skydiscover.search.differentiable.primitives.base import ProgramState, SoftPrimitive
from skydiscover.search.differentiable.primitives.composition import (
    PrimitiveGraph,
    Sequence,
    WeightedChoice,
)
from skydiscover.search.differentiable.primitives.conditions import SoftGT
from skydiscover.search.differentiable.primitives.functions import SoftSwap
from skydiscover.search.differentiable.program_repr import DifferentiableProgram
from skydiscover.search.gradient_pure.database import GradientPureDatabase
from skydiscover.utils.metrics import get_score

logger = logging.getLogger(__name__)


class GradientPureController(DiscoveryController):
    """Pure gradient-based search controller.

    No LLM involvement. Works entirely through gradient descent on
    differentiable primitive graphs. New structures emerge from
    perturbations of WeightedChoice architecture weights.
    """

    def __init__(self, controller_input: DiscoveryControllerInput):
        # Skip LLM initialization from parent - pure gradient doesn't need LLMs.
        # We manually set up only what we need from DiscoveryController.
        import multiprocessing as mp

        from skydiscover.evaluation.evaluator import Evaluator

        self.config = controller_input.config
        self.evaluation_file = controller_input.evaluation_file
        self.database = controller_input.database
        self.file_suffix = controller_input.file_suffix
        self.output_dir = controller_input.output_dir
        self.shutdown_event = mp.Event()
        self.early_stopping_triggered = False
        self.monitor_callback = None
        self.feedback_reader = None
        self.num_context_programs = controller_input.config.search.num_context_programs

        # Set up evaluator (no LLM judge for pure gradient)
        self.config.evaluator.evaluation_file = self.evaluation_file
        self.config.evaluator.file_suffix = self.file_suffix
        self.config.evaluator.is_image_mode = self.config.language == "image"
        self.evaluator = Evaluator(
            self.config.evaluator,
            llm_judge=None,
            max_concurrent=max(self.config.max_parallel_iterations, 4),
        )

        db_config = self.config.search.database

        self.opt_config = OptimizationConfig(
            learning_rate=getattr(db_config, "learning_rate", 0.01),
            optimization_steps=getattr(db_config, "optimization_steps", 50),
            beta_start=getattr(db_config, "beta_start", 1.0),
            beta_end=getattr(db_config, "beta_end", 50.0),
            beta_anneal_steps=getattr(db_config, "beta_anneal_steps", 30),
        )
        self.optimizer = PrimitiveOptimizer(self.opt_config)

        # Perturbation config
        self.perturbation_std = getattr(db_config, "perturbation_std", 0.5)
        self.perturbation_ratio = getattr(db_config, "perturbation_ratio", 0.3)

        # Proxy loss and input generation
        self.proxy_loss_fn = None
        self._generate_inputs_fn = None
        proxy_loss_file = getattr(db_config, "proxy_loss_file", None)
        if proxy_loss_file:
            self._load_proxy_loss_module(proxy_loss_file)

        self.training_inputs = []

        logger.info(
            f"GradientPureController initialized "
            f"(lr={self.opt_config.learning_rate}, "
            f"steps={self.opt_config.optimization_steps})"
        )

    def _load_proxy_loss_module(self, path):
        """Load proxy loss function and optional generate_inputs from file."""
        import importlib.util

        # Resolve path relative to evaluation file directory if not absolute
        if not os.path.isabs(path) and not os.path.exists(path):
            eval_dir = os.path.dirname(os.path.abspath(self.evaluation_file))
            candidate = os.path.join(eval_dir, path)
            if os.path.exists(candidate):
                path = candidate

        try:
            spec = importlib.util.spec_from_file_location("proxy_loss", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "proxy_loss"):
                self.proxy_loss_fn = module.proxy_loss
                logger.info(f"Loaded proxy loss from {path}")
            if hasattr(module, "generate_inputs"):
                self._generate_inputs_fn = module.generate_inputs
                logger.info(f"Loaded generate_inputs from {path}")
        except Exception as e:
            logger.warning(f"Failed to load proxy loss module: {e}")

    def _seed_initial_graphs(self) -> None:
        """Seed the graph population with simple compare-and-swap networks.

        Creates starter PrimitiveGraphs so pure gradient search can begin
        without requiring an LLM to bootstrap.
        """
        db: GradientPureDatabase = self.database
        beta = self.opt_config.beta_start

        # Graph 1: 2-element compare-and-swap
        cond_01 = SoftGT("e0", "e1", beta=beta)
        swap_01 = SoftSwap("e0", "e1", cond_01, beta=beta)
        graph_2 = PrimitiveGraph(
            inputs=["e0", "e1"],
            outputs=["e0", "e1"],
            operations=[swap_01],
            beta=beta,
        )
        db.register_graph(str(uuid.uuid4()), graph_2, 0.0)

        # Graph 2: 3-element bubble sort (compare-swap 01, 12, 01)
        g3_cond_01 = SoftGT("e0", "e1", beta=beta)
        g3_swap_01 = SoftSwap("e0", "e1", g3_cond_01, beta=beta)
        g3_cond_12 = SoftGT("e1", "e2", beta=beta)
        g3_swap_12 = SoftSwap("e1", "e2", g3_cond_12, beta=beta)
        g3_cond_01b = SoftGT("e0", "e1", beta=beta)
        g3_swap_01b = SoftSwap("e0", "e1", g3_cond_01b, beta=beta)
        graph_3 = PrimitiveGraph(
            inputs=["e0", "e1", "e2"],
            outputs=["e0", "e1", "e2"],
            operations=[g3_swap_01, g3_swap_12, g3_swap_01b],
            beta=beta,
        )
        db.register_graph(str(uuid.uuid4()), graph_3, 0.0)

        # Graph 3: 3-element with WeightedChoice over two swap orderings
        # Option A: swap(0,1) then swap(1,2)
        opt_a_c01 = SoftGT("e0", "e1", beta=beta)
        opt_a_s01 = SoftSwap("e0", "e1", opt_a_c01, beta=beta)
        opt_a_c12 = SoftGT("e1", "e2", beta=beta)
        opt_a_s12 = SoftSwap("e1", "e2", opt_a_c12, beta=beta)
        option_a = Sequence([opt_a_s01, opt_a_s12], beta=beta)

        # Option B: swap(1,2) then swap(0,1)
        opt_b_c12 = SoftGT("e1", "e2", beta=beta)
        opt_b_s12 = SoftSwap("e1", "e2", opt_b_c12, beta=beta)
        opt_b_c01 = SoftGT("e0", "e1", beta=beta)
        opt_b_s01 = SoftSwap("e0", "e1", opt_b_c01, beta=beta)
        option_b = Sequence([opt_b_s12, opt_b_s01], beta=beta)

        choice = WeightedChoice([option_a, option_b], beta=beta)
        # Add a final pass to ensure sorted
        final_c01 = SoftGT("e0", "e1", beta=beta)
        final_s01 = SoftSwap("e0", "e1", final_c01, beta=beta)
        graph_choice = PrimitiveGraph(
            inputs=["e0", "e1", "e2"],
            outputs=["e0", "e1", "e2"],
            operations=[choice, final_s01],
            beta=beta,
        )
        db.register_graph(str(uuid.uuid4()), graph_choice, 0.0)

        logger.info(f"Seeded {len(db.graph_population)} initial graphs")

    def _populate_training_inputs(self) -> None:
        """Generate training inputs for gradient optimization."""
        if self._generate_inputs_fn:
            self.training_inputs = [self._generate_inputs_fn() for _ in range(4)]
            logger.info(f"Generated {len(self.training_inputs)} training input batches")
        else:
            # Default: random arrays for element-wise variables
            for _ in range(4):
                state = ProgramState(batch_size=8)
                n = random.choice([3, 4, 5])
                for i in range(n):
                    state[f"e{i}"] = torch.randn(8)
                self.training_inputs.append(state)
            logger.info(f"Generated {len(self.training_inputs)} default training input batches")

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback=None,
    ) -> Optional[Program]:
        """Run pure gradient-based discovery."""
        total = start_iteration + max_iterations
        logger.info(f"Gradient pure search: running {max_iterations} iterations")

        # Bootstrap: seed graphs and training inputs if needed
        if isinstance(self.database, GradientPureDatabase):
            if not self.database.graph_population:
                self._seed_initial_graphs()
            if not self.training_inputs and self.proxy_loss_fn:
                self._populate_training_inputs()

        for iteration in range(start_iteration, total):
            if self.shutdown_event.is_set():
                break

            try:
                iter_start = time.time()

                if not isinstance(self.database, GradientPureDatabase):
                    logger.warning("Database is not GradientPureDatabase, skipping iteration")
                    continue

                db: GradientPureDatabase = self.database

                if not db.graph_population:
                    logger.warning("No graphs in population, skipping iteration")
                    continue

                # Decide: exploit existing or perturb
                should_perturb = random.random() < self.perturbation_ratio

                if should_perturb:
                    result = db.get_best_graph()
                else:
                    result = db.get_random_graph()

                if result is None:
                    logger.warning("Failed to sample graph, skipping iteration")
                    continue

                graph_id, graph = result

                # Clone and optionally perturb
                graph_copy = copy.deepcopy(graph)
                if should_perturb:
                    self._perturb_weights(graph_copy)

                # Gradient optimize
                loss_history = []
                if self.proxy_loss_fn and self.training_inputs:
                    _, loss_history = self.optimizer.optimize(
                        graph_copy, self.proxy_loss_fn, self.training_inputs
                    )

                # Discretize and evaluate
                solution = graph_to_solution(graph_copy)
                program_id = str(uuid.uuid4())
                eval_result = await self.evaluator.evaluate_program(solution, program_id)

                if eval_result is None:
                    continue

                program = DifferentiableProgram(
                    id=program_id,
                    solution=solution,
                    language=self.config.language or "python",
                    metrics=eval_result.metrics if hasattr(eval_result, "metrics") else {},
                    artifacts=eval_result.artifacts if hasattr(eval_result, "artifacts") else {},
                    iteration_found=iteration,
                    optimization_history=loss_history,
                    optimization_steps_used=len(loss_history),
                    mode="gradient_pure",
                )

                db.add(program, iteration=iteration)

                # Register the optimized graph
                new_graph_id = str(uuid.uuid4())
                score = get_score(program.metrics)
                db.register_graph(new_graph_id, graph_copy, score)

                iter_time = time.time() - iter_start
                logger.info(
                    f"Iteration {iteration}: "
                    f"{'perturb' if should_perturb else 'optimize'} "
                    f"(score={score:.4f}, {iter_time:.1f}s)"
                )

                if checkpoint_callback:
                    checkpoint_callback(iteration)

            except Exception as e:
                logger.exception(f"Iteration {iteration} failed: {e}")

        best = self.database.get_best_program()
        if best:
            logger.info(f"Gradient pure search completed. Best: {get_score(best.metrics):.4f}")
        return best

    def _perturb_weights(self, graph: torch.nn.Module) -> None:
        """Add random noise to WeightedChoice architecture weights."""
        for module in graph.modules():
            if isinstance(module, WeightedChoice):
                with torch.no_grad():
                    noise = torch.randn_like(module.arch_weights) * self.perturbation_std
                    module.arch_weights.add_(noise)

    async def _run_llm_fallback(self, iteration: int) -> None:
        """Fallback to standard LLM iteration when no graphs available."""
        parent, context_programs = self.database.sample(self.num_context_programs)

        prompt = self.context_builder.build_prompt(
            parent,
            context={
                "other_context_programs": context_programs,
                "program_metrics": parent.metrics,
            },
        )

        response = await self._call_llm(
            prompt.get("system", ""), prompt.get("user", "")
        )

        if not response or not response.text:
            return

        from skydiscover.utils.code_utils import extract_diffs, apply_diff, parse_full_rewrite

        solution = None
        diffs = extract_diffs(response.text)
        if diffs:
            solution = apply_diff(parent.solution, diffs)
        if not solution:
            solution = parse_full_rewrite(response.text, parent.solution)
        if not solution:
            solution = response.text

        program_id = str(uuid.uuid4())
        eval_result = await self.evaluator.evaluate_program(solution, program_id)
        if eval_result is None:
            return

        program = DifferentiableProgram(
            id=program_id,
            solution=solution,
            language=self.config.language or "python",
            metrics=eval_result.metrics if hasattr(eval_result, "metrics") else {},
            artifacts=eval_result.artifacts if hasattr(eval_result, "artifacts") else {},
            iteration_found=iteration,
            parent_id=parent.id,
            generation=parent.generation + 1,
            mode="gradient_pure",
        )
        self.database.add(program, iteration=iteration)
