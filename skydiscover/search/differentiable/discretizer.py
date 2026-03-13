"""
Discretizer: converts a soft PrimitiveGraph to hard (discrete) mode
and generates executable code.

The soft->hard transition:
1. Set hard=True on all primitives
2. WeightedChoice nodes -> argmax selects one option
3. Conditions -> snap at 0.5 threshold
4. Generate code via to_code()
"""

from __future__ import annotations

import logging

import torch.nn as nn

from skydiscover.search.differentiable.primitives.base import SoftPrimitive

logger = logging.getLogger(__name__)


def discretize_graph(graph: nn.Module) -> str:
    """Convert a soft PrimitiveGraph to executable Python code.

    Steps:
    1. Set hard=True on all primitives (makes conditions use thresholds)
    2. Call to_code() to generate the discrete code string

    Args:
        graph: A PrimitiveGraph or SoftPrimitive module.

    Returns:
        Executable Python code string.
    """
    # Set all primitives to hard mode
    if hasattr(graph, "set_hard_recursive"):
        graph.set_hard_recursive(True)
    else:
        _set_hard_recursive(graph, True)

    # Generate code
    if hasattr(graph, "to_code"):
        code = graph.to_code(indent=0)
    else:
        logger.warning("Graph has no to_code method, returning empty string")
        code = ""

    # Reset to soft mode for potential further optimization
    if hasattr(graph, "set_hard_recursive"):
        graph.set_hard_recursive(False)
    else:
        _set_hard_recursive(graph, False)

    return code


def _set_hard_recursive(module: nn.Module, hard: bool) -> None:
    """Recursively set hard mode on all SoftPrimitive children."""
    if isinstance(module, SoftPrimitive):
        module.hard = hard
    for child in module.children():
        _set_hard_recursive(child, hard)
