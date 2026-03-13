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
from skydiscover.search.differentiable.primitives.base import SoftPrimitive
from skydiscover.search.differentiable.primitives.composition import WeightedChoice
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
        super().__init__(controller_input)

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

        # Proxy loss
        self.proxy_loss_fn = None
        proxy_loss_file = getattr(db_config, "proxy_loss_file", None)
        if proxy_loss_file:
            self.proxy_loss_fn = self._load_proxy_loss(proxy_loss_file)

        self.training_inputs = []

        logger.info(
            f"GradientPureController initialized "
            f"(lr={self.opt_config.learning_rate}, "
            f"steps={self.opt_config.optimization_steps})"
        )

    def _load_proxy_loss(self, path):
        """Load proxy loss function from file."""
        import importlib.util

        try:
            spec = importlib.util.spec_from_file_location("proxy_loss", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "proxy_loss"):
                return module.proxy_loss
        except Exception as e:
            logger.warning(f"Failed to load proxy loss: {e}")
        return None

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback=None,
    ) -> Optional[Program]:
        """Run pure gradient-based discovery."""
        total = start_iteration + max_iterations
        logger.info(f"Gradient pure search: running {max_iterations} iterations")

        for iteration in range(start_iteration, total):
            if self.shutdown_event.is_set():
                break

            try:
                iter_start = time.time()

                if not isinstance(self.database, GradientPureDatabase):
                    # Fallback to LLM iteration
                    await self._run_llm_fallback(iteration)
                    continue

                db: GradientPureDatabase = self.database

                if not db.graph_population:
                    # No graphs yet - use LLM to bootstrap
                    await self._run_llm_fallback(iteration)
                    continue

                # Decide: exploit existing or perturb
                should_perturb = random.random() < self.perturbation_ratio

                if should_perturb:
                    result = db.get_best_graph()
                else:
                    result = db.get_random_graph()

                if result is None:
                    await self._run_llm_fallback(iteration)
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
