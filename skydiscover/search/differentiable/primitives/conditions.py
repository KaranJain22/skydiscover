"""
Soft (differentiable) comparison operators.

Ported from AlgoVision's conditions.py. Uses continuous relaxations:
- GT/LT: sigmoid-based
- Eq/NEq: hyperbolic secant-based

During search (hard=False): returns soft probability in [0, 1].
During evaluation (hard=True): returns hard 0/1.
"""

from __future__ import annotations

from typing import Union

import torch

from skydiscover.search.differentiable.primitives.base import (
    ProgramState,
    SoftPrimitive,
)


class SoftCondition(SoftPrimitive):
    """Base class for soft conditions that compare two variables.

    Conditions return a probability tensor of shape (batch_size,).
    """

    def __init__(
        self,
        left: str,
        right: Union[str, float],
        beta: float = 10.0,
        hard: bool = False,
    ):
        super().__init__(beta=beta, hard=hard)
        self.left = left
        self.right = right

    def _get_left(self, state: ProgramState) -> torch.Tensor:
        return state[self.left]

    def _get_right(self, state: ProgramState) -> Union[torch.Tensor, float]:
        if isinstance(self.right, str):
            return state[self.right]
        return self.right

    def _right_code(self) -> str:
        if isinstance(self.right, str):
            return self.right
        return repr(self.right)


class SoftGT(SoftCondition):
    """Soft greater-than: sigmoid(beta * (left - right)).

    Ported from AlgoVision GT (conditions.py:42-49).
    """

    def forward(self, state: ProgramState) -> torch.Tensor:
        diff = self._get_left(state) - self._get_right(state)
        if self.hard:
            return (diff > 0).float()
        return torch.sigmoid(self.beta * diff)

    def to_code(self, indent: int = 0) -> str:
        return f"{self.left} > {self._right_code()}"


class SoftLT(SoftCondition):
    """Soft less-than: sigmoid(-beta * (left - right)).

    Ported from AlgoVision LT (conditions.py:29-37).
    """

    def forward(self, state: ProgramState) -> torch.Tensor:
        diff = self._get_left(state) - self._get_right(state)
        if self.hard:
            return (diff < 0).float()
        return torch.sigmoid(-self.beta * diff)

    def to_code(self, indent: int = 0) -> str:
        return f"{self.left} < {self._right_code()}"


class SoftEq(SoftCondition):
    """Soft equality: 1/cosh(beta/2 * (left - right))^2.

    Ported from AlgoVision Eq (conditions.py:9-16).
    """

    def forward(self, state: ProgramState) -> torch.Tensor:
        diff = self._get_left(state) - self._get_right(state)
        if self.hard:
            return (diff.abs() < 1e-6).float()
        diff = diff * self.beta / 2.0
        return 1.0 / torch.cosh(diff) ** 2

    def to_code(self, indent: int = 0) -> str:
        return f"{self.left} == {self._right_code()}"


class SoftNEq(SoftCondition):
    """Soft not-equal: 1 - 1/cosh(beta/2 * (left - right))^2.

    Ported from AlgoVision NEq (conditions.py:19-26).
    """

    def forward(self, state: ProgramState) -> torch.Tensor:
        diff = self._get_left(state) - self._get_right(state)
        if self.hard:
            return (diff.abs() >= 1e-6).float()
        diff = diff * self.beta / 2.0
        return 1.0 - 1.0 / torch.cosh(diff) ** 2

    def to_code(self, indent: int = 0) -> str:
        return f"{self.left} != {self._right_code()}"
