"""
Differentiable primitives library for gradient-based algorithm discovery.

Provides soft (continuous) relaxations of discrete algorithmic operations.
During search, primitives operate in soft mode for gradient flow.
During evaluation, primitives are discretized to produce executable code.
"""

from skydiscover.search.differentiable.primitives.base import (
    ProgramState,
    SoftPrimitive,
)
from skydiscover.search.differentiable.primitives.conditions import (
    SoftEq,
    SoftGT,
    SoftLT,
    SoftNEq,
)
from skydiscover.search.differentiable.primitives.control_flow import (
    SoftFor,
    SoftIf,
    SoftWhile,
)
from skydiscover.search.differentiable.primitives.functions import (
    SoftArgMax,
    SoftArgMin,
    SoftMax,
    SoftMin,
    SoftSelect,
    SoftSwap,
)
from skydiscover.search.differentiable.primitives.composition import (
    PrimitiveGraph,
    Sequence,
    WeightedChoice,
)

__all__ = [
    "ProgramState",
    "SoftPrimitive",
    "SoftGT",
    "SoftLT",
    "SoftEq",
    "SoftNEq",
    "SoftIf",
    "SoftWhile",
    "SoftFor",
    "SoftMin",
    "SoftMax",
    "SoftArgMin",
    "SoftArgMax",
    "SoftSwap",
    "SoftSelect",
    "PrimitiveGraph",
    "Sequence",
    "WeightedChoice",
]
