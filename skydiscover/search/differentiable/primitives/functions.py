"""
Soft (differentiable) functions: min, max, argmin, argmax, swap, select.

Ported from AlgoVision's functions.py. Uses softmax/softmin with
inverse temperature beta for continuous relaxation.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from skydiscover.search.differentiable.primitives.base import (
    ProgramState,
    SoftPrimitive,
)


class SoftMin(SoftPrimitive):
    """Soft minimum via Softmin-weighted sum.

    Ported from AlgoVision Min (functions.py:9-24).
    """

    def __init__(self, var_a: str, var_b: str, result_var: str, beta: float = 10.0, hard: bool = False):
        super().__init__(beta=beta, hard=hard)
        self.var_a = var_a
        self.var_b = var_b
        self.result_var = result_var

    def forward(self, state: ProgramState) -> ProgramState:
        a, b = state[self.var_a], state[self.var_b]
        tensors = torch.stack([a, b], dim=-1)
        if self.hard:
            state[self.result_var] = torch.min(a, b)
        else:
            weights = torch.nn.Softmin(dim=-1)(self.beta * tensors)
            state[self.result_var] = (weights * tensors).sum(dim=-1)
        return state

    def to_code(self, indent: int = 0) -> str:
        return self._indent(f"{self.result_var} = min({self.var_a}, {self.var_b})", indent)


class SoftMax(SoftPrimitive):
    """Soft maximum via Softmax-weighted sum.

    Ported from AlgoVision Max (functions.py:45-60).
    """

    def __init__(self, var_a: str, var_b: str, result_var: str, beta: float = 10.0, hard: bool = False):
        super().__init__(beta=beta, hard=hard)
        self.var_a = var_a
        self.var_b = var_b
        self.result_var = result_var

    def forward(self, state: ProgramState) -> ProgramState:
        a, b = state[self.var_a], state[self.var_b]
        tensors = torch.stack([a, b], dim=-1)
        if self.hard:
            state[self.result_var] = torch.max(a, b)
        else:
            weights = torch.nn.Softmax(dim=-1)(self.beta * tensors)
            state[self.result_var] = (weights * tensors).sum(dim=-1)
        return state

    def to_code(self, indent: int = 0) -> str:
        return self._indent(f"{self.result_var} = max({self.var_a}, {self.var_b})", indent)


class SoftArgMin(SoftPrimitive):
    """Soft argmin returning probability distribution.

    Ported from AlgoVision ArgMin (functions.py:27-42).
    """

    def __init__(self, var_a: str, var_b: str, result_var: str, beta: float = 10.0, hard: bool = False):
        super().__init__(beta=beta, hard=hard)
        self.var_a = var_a
        self.var_b = var_b
        self.result_var = result_var

    def forward(self, state: ProgramState) -> ProgramState:
        a, b = state[self.var_a], state[self.var_b]
        tensors = torch.stack([a, b], dim=-1)
        if self.hard:
            state[self.result_var] = (a > b).float()  # 0 if a is min, 1 if b is min
        else:
            state[self.result_var] = torch.nn.Softmin(dim=-1)(self.beta * tensors)
        return state

    def to_code(self, indent: int = 0) -> str:
        return self._indent(
            f"{self.result_var} = 0 if {self.var_a} <= {self.var_b} else 1", indent
        )


class SoftArgMax(SoftPrimitive):
    """Soft argmax returning probability distribution.

    Ported from AlgoVision ArgMax (functions.py:63-78).
    """

    def __init__(self, var_a: str, var_b: str, result_var: str, beta: float = 10.0, hard: bool = False):
        super().__init__(beta=beta, hard=hard)
        self.var_a = var_a
        self.var_b = var_b
        self.result_var = result_var

    def forward(self, state: ProgramState) -> ProgramState:
        a, b = state[self.var_a], state[self.var_b]
        tensors = torch.stack([a, b], dim=-1)
        if self.hard:
            state[self.result_var] = (a < b).float()  # 0 if a is max, 1 if b is max
        else:
            state[self.result_var] = torch.nn.Softmax(dim=-1)(self.beta * tensors)
        return state

    def to_code(self, indent: int = 0) -> str:
        return self._indent(
            f"{self.result_var} = 0 if {self.var_a} >= {self.var_b} else 1", indent
        )


class SoftSwap(SoftPrimitive):
    """Differentiable conditional swap of two variables.

    If condition is true (soft probability p): swap var_a and var_b.
    In soft mode: var_a' = p*var_b + (1-p)*var_a, var_b' = p*var_a + (1-p)*var_b
    """

    def __init__(
        self,
        var_a: str,
        var_b: str,
        condition: SoftPrimitive,
        beta: float = 10.0,
        hard: bool = False,
    ):
        super().__init__(beta=beta, hard=hard)
        self.var_a = var_a
        self.var_b = var_b
        self.condition = condition
        if isinstance(condition, nn.Module):
            self.add_module("swap_condition", condition)

    def forward(self, state: ProgramState) -> ProgramState:
        p = self.condition(state)
        a = state[self.var_a]
        b = state[self.var_b]

        if self.hard:
            mask = (p > 0.5).float()
        else:
            mask = p

        while len(mask.shape) < len(a.shape):
            mask = mask.unsqueeze(-1)

        state[self.var_a] = mask * b + (1 - mask) * a
        state[self.var_b] = mask * a + (1 - mask) * b
        return state

    def to_code(self, indent: int = 0) -> str:
        cond_code = self.condition.to_code() if hasattr(self.condition, "to_code") else "True"
        lines = [
            self._indent(f"if {cond_code}:", indent),
            self._indent(f"{self.var_a}, {self.var_b} = {self.var_b}, {self.var_a}", indent + 1),
        ]
        return "\n".join(lines)


class SoftSelect(SoftPrimitive):
    """Differentiable selection over multiple options using learnable weights.

    Weights are nn.Parameters optimized by gradient descent.
    During discretization: argmax(weights) selects one option.
    """

    def __init__(
        self,
        options: List[SoftPrimitive],
        result_var: str,
        beta: float = 10.0,
        hard: bool = False,
    ):
        super().__init__(beta=beta, hard=hard)
        self.options = nn.ModuleList(options)
        self.result_var = result_var
        self.weights = nn.Parameter(torch.zeros(len(options)))

    def forward(self, state: ProgramState) -> ProgramState:
        if self.hard:
            idx = self.weights.argmax().item()
            return self.options[idx](state)

        # Soft: weighted combination of all option outputs
        probs = torch.softmax(self.beta * self.weights, dim=0)
        states = []
        for option in self.options:
            option_state = state.clone()
            option_state = option(option_state)
            states.append(option_state)

        # Merge all states weighted by probs
        result_state = states[0]
        for key in result_state.keys():
            val = result_state[key]
            if isinstance(val, torch.Tensor):
                weighted = probs[0] * val
                for i in range(1, len(states)):
                    other_val = states[i][key]
                    if isinstance(other_val, torch.Tensor):
                        weighted = weighted + probs[i] * other_val
                result_state[key] = weighted

        return result_state

    def to_code(self, indent: int = 0) -> str:
        idx = self.weights.argmax().item()
        return self.options[idx].to_code(indent)
