"""
DifferentiableController: hybrid search combining LLM structure proposals
with gradient-based parameter optimization.

The hybrid loop:
1. LLM proposes algorithm structures as compositions of primitives (30% of iterations)
2. Gradient descent optimizes parameters within existing structures (70%)
3. Discretize the optimized graph to executable code
4. Evaluate via standard SkyDiscover evaluator
5. Store results in database
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from skydiscover.context_builder.default import DefaultContextBuilder
from skydiscover.evaluation.evaluator import Evaluator
from skydiscover.evaluation.llm_judge import LLMJudge
from skydiscover.llm.llm_pool import LLMPool
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.differentiable.codegen import graph_to_solution
from skydiscover.search.differentiable.database import DifferentiableDatabase
from skydiscover.search.differentiable.discretizer import discretize_graph
from skydiscover.search.differentiable.optimizer import (
    OptimizationConfig,
    PrimitiveOptimizer,
)
from skydiscover.search.differentiable.program_repr import DifferentiableProgram
from skydiscover.utils.metrics import get_score

logger = logging.getLogger(__name__)


class DifferentiableController(DiscoveryController):
    """Hybrid differentiable search controller.

    Combines LLM-based structural search with gradient-based parameter
    optimization. The LLM proposes algorithm structures as compositions
    of differentiable primitives; gradient descent optimizes the continuous
    parameters within those structures; evaluation discretizes via
    beta-annealing.
    """

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)

        db_config = self.config.search.database

        # Gradient optimization config
        self.opt_config = OptimizationConfig(
            learning_rate=getattr(db_config, "learning_rate", 0.01),
            optimization_steps=getattr(db_config, "optimization_steps", 50),
            beta_start=getattr(db_config, "beta_start", 1.0),
            beta_end=getattr(db_config, "beta_end", 50.0),
            beta_anneal_steps=getattr(db_config, "beta_anneal_steps", 30),
        )
        self.optimizer = PrimitiveOptimizer(self.opt_config)

        # Structure vs parameter ratio
        self.structure_proposal_ratio = getattr(db_config, "structure_proposal_ratio", 0.3)

        # Proxy loss
        self.proxy_loss_fn: Optional[Callable] = None
        proxy_loss_file = getattr(db_config, "proxy_loss_file", None)
        if proxy_loss_file:
            self.proxy_loss_fn = self._load_proxy_loss(proxy_loss_file)

        # Training data for gradient optimization
        self.training_inputs: List[Dict[str, torch.Tensor]] = []

        logger.info(
            f"DifferentiableController initialized "
            f"(structure_ratio={self.structure_proposal_ratio}, "
            f"lr={self.opt_config.learning_rate}, "
            f"steps={self.opt_config.optimization_steps})"
        )

    def _load_proxy_loss(self, path: str) -> Optional[Callable]:
        """Load a user-provided differentiable proxy loss function.

        The file should define a function: proxy_loss(state: ProgramState) -> Tensor
        """
        try:
            path = os.path.abspath(path)
            spec = importlib.util.spec_from_file_location("proxy_loss_module", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "proxy_loss"):
                logger.info(f"Loaded proxy loss from {path}")
                return module.proxy_loss
            else:
                logger.warning(f"No 'proxy_loss' function found in {path}")
        except Exception as e:
            logger.warning(f"Failed to load proxy loss from {path}: {e}")
        return None

    # ------------------------------------------------------------------
    # Main discovery loop
    # ------------------------------------------------------------------

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback=None,
    ) -> Optional[Program]:
        """Run hybrid differentiable discovery."""
        total = start_iteration + max_iterations
        logger.info(
            f"Differentiable search: running {max_iterations} iterations "
            f"(structure_ratio={self.structure_proposal_ratio})"
        )

        for iteration in range(start_iteration, total):
            if self.shutdown_event.is_set():
                logger.info("Shutdown requested")
                break

            try:
                iter_start = time.time()

                # Decide: propose new structure or reoptimize existing
                should_propose = self._should_propose_structure(iteration)

                if should_propose or not self._has_optimizable_structures():
                    # LLM proposes a new structure -> evaluate without gradient opt
                    await self._run_llm_iteration(iteration)
                else:
                    # Gradient optimization on existing structure
                    await self._run_gradient_iteration(iteration)

                iter_time = time.time() - iter_start
                logger.info(
                    f"Iteration {iteration}: "
                    f"{'structure proposal' if should_propose else 'gradient opt'} "
                    f"({iter_time:.1f}s)"
                )

                if checkpoint_callback:
                    checkpoint_callback(iteration)

            except Exception as e:
                logger.exception(f"Iteration {iteration} failed: {e}")

        best = self.database.get_best_program()
        if best:
            logger.info(
                f"Differentiable search completed. Best score: {get_score(best.metrics):.4f}"
            )
        return best

    def _should_propose_structure(self, iteration: int) -> bool:
        """Decide whether to propose a new structure vs reoptimize."""
        import random

        # Always propose for first few iterations
        if len(self.database.programs) < 3:
            return True
        return random.random() < self.structure_proposal_ratio

    def _has_optimizable_structures(self) -> bool:
        """Check if we have structures with graphs for reoptimization."""
        if not isinstance(self.database, DifferentiableDatabase):
            return False
        return self.database.get_structure_for_reoptimization() is not None

    # ------------------------------------------------------------------
    # LLM-based structure proposal (reuses standard SkyDiscover pipeline)
    # ------------------------------------------------------------------

    async def _run_llm_iteration(self, iteration: int) -> None:
        """Use LLM to propose a new algorithm (standard evolution step).

        This is essentially the same as the default discovery controller
        iteration but creates a DifferentiableProgram.
        """
        parent, context_programs = self.database.sample(self.num_context_programs)

        # Build prompt
        prompt = self.context_builder.build_prompt(
            parent,
            context={
                "other_context_programs": context_programs,
                "program_metrics": parent.metrics,
            },
        )

        # Generate via LLM
        response = await self._call_llm(
            prompt.get("system", ""),
            prompt.get("user", ""),
        )

        if not response or not response.text:
            logger.warning(f"Empty LLM response at iteration {iteration}")
            return

        # Parse response into solution code
        from skydiscover.utils.code_utils import extract_diffs, apply_diff, parse_full_rewrite

        solution = None
        diffs = extract_diffs(response.text)
        if diffs:
            solution = apply_diff(parent.solution, diffs)
        if not solution:
            solution = parse_full_rewrite(response.text, parent.solution)
        if not solution:
            solution = response.text

        # Evaluate
        program_id = str(uuid.uuid4())
        eval_result = await self.evaluator.evaluate_program(solution, program_id)

        if eval_result is None:
            logger.warning(f"Evaluation failed at iteration {iteration}")
            return

        # Create DifferentiableProgram
        program = DifferentiableProgram(
            id=program_id,
            solution=solution,
            language=self.config.language or "python",
            metrics=eval_result.metrics if hasattr(eval_result, "metrics") else {},
            artifacts=eval_result.artifacts if hasattr(eval_result, "artifacts") else {},
            iteration_found=iteration,
            parent_id=parent.id,
            generation=parent.generation + 1,
            mode="hybrid",
        )

        self.database.add(program, iteration=iteration)

    # ------------------------------------------------------------------
    # Gradient-based parameter optimization
    # ------------------------------------------------------------------

    async def _run_gradient_iteration(self, iteration: int) -> None:
        """Reoptimize parameters of an existing structure via gradients.

        1. Get the best structure from the database
        2. Run gradient optimization with beta annealing
        3. Discretize to code
        4. Evaluate the discretized code
        5. Add result to database
        """
        if not isinstance(self.database, DifferentiableDatabase):
            await self._run_llm_iteration(iteration)
            return

        result = self.database.get_structure_for_reoptimization()
        if result is None:
            await self._run_llm_iteration(iteration)
            return

        structure_hash, graph = result

        # Clone the graph for this optimization run
        import copy

        graph_copy = copy.deepcopy(graph)

        # Run gradient optimization if proxy loss is available
        loss_history = []
        if self.proxy_loss_fn and self.training_inputs:
            optimized_params, loss_history = self.optimizer.optimize(
                graph_copy,
                self.proxy_loss_fn,
                self.training_inputs,
            )

        # Discretize to code
        solution = graph_to_solution(graph_copy)

        # Evaluate
        program_id = str(uuid.uuid4())
        eval_result = await self.evaluator.evaluate_program(solution, program_id)

        if eval_result is None:
            logger.warning(f"Gradient iteration {iteration}: evaluation failed")
            return

        # Create program
        program = DifferentiableProgram(
            id=program_id,
            solution=solution,
            language=self.config.language or "python",
            metrics=eval_result.metrics if hasattr(eval_result, "metrics") else {},
            artifacts=eval_result.artifacts if hasattr(eval_result, "artifacts") else {},
            iteration_found=iteration,
            structure_hash=structure_hash,
            optimization_history=loss_history,
            beta_final=self.opt_config.beta_end,
            optimization_steps_used=len(loss_history),
            mode="hybrid",
        )

        self.database.add(program, iteration=iteration)
