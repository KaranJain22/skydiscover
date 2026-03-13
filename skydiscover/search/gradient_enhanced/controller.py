"""
GradientEnhancedController: LLM evolution + gradient-based local refinement.

The standard SkyDiscover evolution loop, enhanced with an optional gradient
refinement step after LLM generates a candidate:
1. Sample parent from database
2. LLM generates candidate code
3. (Optional) Parse code into PrimitiveGraph if possible
4. (Optional) Gradient refine continuous parameters
5. Discretize refined version (or use original if parsing failed)
6. Evaluate
7. Add to database
"""

from __future__ import annotations

import copy
import logging
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
from skydiscover.search.differentiable.program_repr import DifferentiableProgram
from skydiscover.search.gradient_enhanced.database import GradientEnhancedDatabase
from skydiscover.utils.metrics import get_score

logger = logging.getLogger(__name__)


class GradientEnhancedController(DiscoveryController):
    """Evolution controller with gradient-based local refinement.

    Uses standard LLM evolution as the primary search strategy.
    After generating a candidate, attempts to gradient-refine its
    continuous parameters before evaluation. Falls back gracefully
    if the candidate can't be parsed into a differentiable graph.
    """

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)

        db_config = self.config.search.database

        # Gradient refinement config (fewer steps than pure gradient)
        self.opt_config = OptimizationConfig(
            learning_rate=getattr(db_config, "learning_rate", 0.01),
            optimization_steps=getattr(db_config, "refinement_steps", 20),
            beta_start=getattr(db_config, "beta_start", 5.0),
            beta_end=getattr(db_config, "beta_end", 50.0),
            beta_anneal_steps=getattr(db_config, "beta_anneal_steps", 15),
        )
        self.optimizer = PrimitiveOptimizer(self.opt_config)

        # Whether to always try gradient refinement
        self.enable_refinement = getattr(db_config, "enable_gradient_refinement", True)

        # Proxy loss
        self.proxy_loss_fn = None
        proxy_loss_file = getattr(db_config, "proxy_loss_file", None)
        if proxy_loss_file:
            import importlib.util
            try:
                spec = importlib.util.spec_from_file_location("proxy_loss", proxy_loss_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "proxy_loss"):
                    self.proxy_loss_fn = module.proxy_loss
            except Exception as e:
                logger.warning(f"Failed to load proxy loss: {e}")

        self.training_inputs = []

        logger.info(
            f"GradientEnhancedController initialized "
            f"(refinement={'on' if self.enable_refinement else 'off'}, "
            f"steps={self.opt_config.optimization_steps})"
        )

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback=None,
    ) -> Optional[Program]:
        """Run gradient-enhanced evolution."""
        total = start_iteration + max_iterations
        logger.info(f"Gradient-enhanced evolution: running {max_iterations} iterations")

        for iteration in range(start_iteration, total):
            if self.shutdown_event.is_set():
                break

            try:
                iter_start = time.time()

                # Standard LLM evolution step
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
                    continue

                # Parse response
                from skydiscover.utils.code_utils import (
                    extract_diffs,
                    apply_diff,
                    parse_full_rewrite,
                )

                solution = None
                diffs = extract_diffs(response.text)
                if diffs:
                    solution = apply_diff(parent.solution, diffs)
                if not solution:
                    solution = parse_full_rewrite(response.text, parent.solution)
                if not solution:
                    solution = response.text

                # Evaluate original solution
                program_id = str(uuid.uuid4())
                eval_result = await self.evaluator.evaluate_program(solution, program_id)

                if eval_result is None:
                    continue

                original_score = get_score(
                    eval_result.metrics if hasattr(eval_result, "metrics") else {}
                )

                # Try gradient refinement if enabled and proxy loss available
                gradient_refined = False
                final_solution = solution
                final_metrics = eval_result.metrics if hasattr(eval_result, "metrics") else {}
                final_artifacts = eval_result.artifacts if hasattr(eval_result, "artifacts") else {}

                if (
                    self.enable_refinement
                    and self.proxy_loss_fn
                    and self.training_inputs
                ):
                    refined = await self._try_gradient_refinement(
                        solution, iteration
                    )
                    if refined is not None:
                        refined_solution, refined_score, refined_metrics, refined_artifacts = refined
                        if refined_score > original_score:
                            final_solution = refined_solution
                            final_metrics = refined_metrics
                            final_artifacts = refined_artifacts
                            gradient_refined = True
                            logger.info(
                                f"Gradient refinement improved: "
                                f"{original_score:.4f} -> {refined_score:.4f}"
                            )

                # Create program
                program = DifferentiableProgram(
                    id=program_id,
                    solution=final_solution,
                    language=self.config.language or "python",
                    metrics=final_metrics,
                    artifacts=final_artifacts,
                    iteration_found=iteration,
                    parent_id=parent.id,
                    generation=parent.generation + 1,
                    mode="gradient_enhanced",
                )

                self.database.add(
                    program,
                    iteration=iteration,
                    gradient_refined=gradient_refined,
                )

                iter_time = time.time() - iter_start
                score = get_score(final_metrics)
                logger.info(
                    f"Iteration {iteration}: score={score:.4f} "
                    f"({'refined' if gradient_refined else 'original'}) "
                    f"({iter_time:.1f}s)"
                )

                if checkpoint_callback:
                    checkpoint_callback(iteration)

            except Exception as e:
                logger.exception(f"Iteration {iteration} failed: {e}")

        best = self.database.get_best_program()
        if best:
            logger.info(
                f"Gradient-enhanced evolution completed. Best: {get_score(best.metrics):.4f}"
            )
        return best

    async def _try_gradient_refinement(
        self, solution: str, iteration: int
    ) -> Optional[tuple]:
        """Attempt to gradient-refine a solution.

        This is a placeholder that would parse the solution into a
        PrimitiveGraph, optimize, discretize, and evaluate.

        For the MVP, this attempts to find numerical constants in
        the code and optimize them via gradient descent through the
        proxy loss.

        Returns:
            (refined_solution, score, metrics, artifacts) or None if failed.
        """
        # TODO: Implement code -> PrimitiveGraph parsing for full support.
        # For now, this serves as the integration point.
        logger.debug(f"Gradient refinement not yet implemented for code parsing")
        return None
