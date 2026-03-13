"""
Gradient-based optimizer for differentiable primitive graphs.

Handles:
- Beta annealing: smooth (low beta) -> sharp (high beta) over optimization steps
- Adam optimization of all learnable parameters
- Loss computation via user-provided proxy loss or score predictor
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from skydiscover.search.differentiable.primitives.base import ProgramState

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for gradient optimization."""

    learning_rate: float = 0.01
    optimization_steps: int = 50
    beta_start: float = 1.0
    beta_end: float = 50.0
    beta_anneal_steps: int = 30
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-6


class PrimitiveOptimizer:
    """Optimizes parameters of a differentiable PrimitiveGraph.

    The optimization loop:
    1. Anneals beta from beta_start (smooth) to beta_end (sharp)
    2. Forward pass through soft graph
    3. Compute loss via proxy loss function
    4. Backward pass + gradient clipping + Adam step
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config

    def optimize(
        self,
        graph: nn.Module,
        proxy_loss_fn: Callable[[ProgramState], torch.Tensor],
        training_inputs: List[Dict[str, torch.Tensor]],
        batch_size: int = 32,
    ) -> Tuple[Dict[str, torch.Tensor], List[float]]:
        """Optimize the graph's parameters using gradient descent.

        Args:
            graph: The PrimitiveGraph (nn.Module) to optimize.
            proxy_loss_fn: Differentiable loss function: state -> scalar loss.
            training_inputs: List of input dicts to create ProgramState from.
            batch_size: Batch size for optimization.

        Returns:
            (optimized_params, loss_history)
        """
        params = list(graph.parameters())
        if not params:
            logger.warning("No learnable parameters in graph, skipping optimization")
            return {}, []

        optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)
        loss_history: List[float] = []
        best_loss = float("inf")
        patience_counter = 0

        for step in range(self.config.optimization_steps):
            # Anneal beta
            if step < self.config.beta_anneal_steps:
                t = step / max(self.config.beta_anneal_steps - 1, 1)
                beta = self.config.beta_start + (self.config.beta_end - self.config.beta_start) * t
            else:
                beta = self.config.beta_end

            graph.set_beta_recursive(beta) if hasattr(graph, "set_beta_recursive") else None

            total_loss = torch.tensor(0.0)
            num_batches = 0

            for input_data in training_inputs:
                # Create program state from inputs
                if isinstance(input_data, ProgramState):
                    state = input_data.clone()
                else:
                    state = ProgramState(batch_size=batch_size)
                    for key, value in input_data.items():
                        if isinstance(value, torch.Tensor):
                            state[key] = value
                        else:
                            state[key] = torch.tensor(value, dtype=torch.float32)

                # Forward pass
                output_state = graph(state)

                # Compute loss
                loss = proxy_loss_fn(output_state)
                total_loss = total_loss + loss

                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)

            # Backward pass
            optimizer.zero_grad()
            avg_loss.backward()

            # Gradient clipping
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(params, self.config.grad_clip_norm)

            optimizer.step()

            loss_val = avg_loss.item()
            loss_history.append(loss_val)

            # Early stopping check
            if loss_val < best_loss - self.config.early_stop_min_delta:
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stop_patience:
                logger.debug(f"Early stopping at step {step} (loss={loss_val:.6f})")
                break

        # Collect optimized parameters
        optimized = {name: p.data.clone() for name, p in graph.named_parameters()}

        logger.debug(
            f"Optimization completed: {len(loss_history)} steps, "
            f"loss {loss_history[0]:.6f} -> {loss_history[-1]:.6f}"
        )

        return optimized, loss_history
