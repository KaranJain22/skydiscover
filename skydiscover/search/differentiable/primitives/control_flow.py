"""
Soft (differentiable) control flow structures.

Ported from AlgoVision's control_structures.py:
- SoftIf: executes both branches, merges probabilistically
- SoftWhile: accumulates state weighted by continuation probability
- SoftFor: standard loop (no relaxation needed for fixed ranges)
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch

from skydiscover.search.differentiable.primitives.base import (
    ProgramState,
    SoftPrimitive,
)
from skydiscover.search.differentiable.primitives.conditions import SoftCondition


class SoftIf(SoftPrimitive):
    """Differentiable if-else that executes both branches and merges.

    Ported from AlgoVision If (control_structures.py:27-61).

    In soft mode: state = p * state_true + (1-p) * state_false
    In hard mode: state = (p>0.5) * state_true + (p<=0.5) * state_false
    """

    def __init__(
        self,
        condition: SoftCondition,
        if_true: Optional[Union[SoftPrimitive, List[SoftPrimitive]]] = None,
        if_false: Optional[Union[SoftPrimitive, List[SoftPrimitive]]] = None,
        epsilon: float = 1e-5,
        beta: float = 10.0,
        hard: bool = False,
    ):
        super().__init__(beta=beta, hard=hard)
        self.condition = condition
        self.if_true = if_true if isinstance(if_true, list) else ([if_true] if if_true else [])
        self.if_false = if_false if isinstance(if_false, list) else ([if_false] if if_false else [])
        self.epsilon = epsilon

        # Register child modules
        for i, mod in enumerate(self.if_true):
            if isinstance(mod, SoftPrimitive):
                self.add_module(f"if_true_{i}", mod)
        for i, mod in enumerate(self.if_false):
            if isinstance(mod, SoftPrimitive):
                self.add_module(f"if_false_{i}", mod)
        if isinstance(condition, SoftPrimitive):
            self.add_module("condition_mod", condition)

    def forward(self, state: ProgramState) -> ProgramState:
        p = self.condition(state)

        state_true = state
        state_false = state.clone()

        # Execute true branch
        if self.if_true and (p > self.epsilon).any():
            for mod in self.if_true:
                state_true = mod(state_true)

        # Execute false branch
        if self.if_false and ((1 - p) > self.epsilon).any():
            for mod in self.if_false:
                state_false = mod(state_false)

        # Merge
        if not self.hard:
            state_true.merge(state_false, 1 - p)
        else:
            state_true.merge(state_false, 1 - (p > 0.5).float())

        return state_true

    def to_code(self, indent: int = 0) -> str:
        lines = []
        cond_code = self.condition.to_code()
        lines.append(self._indent(f"if {cond_code}:", indent))
        if self.if_true:
            for mod in self.if_true:
                lines.append(mod.to_code(indent + 1))
        else:
            lines.append(self._indent("pass", indent + 1))
        if self.if_false:
            lines.append(self._indent("else:", indent))
            for mod in self.if_false:
                lines.append(mod.to_code(indent + 1))
        return "\n".join(lines)


class SoftWhile(SoftPrimitive):
    """Differentiable while loop with probabilistic termination.

    Ported from AlgoVision While (control_structures.py:82-120).

    Accumulates state contributions weighted by the probability of
    reaching each iteration. Uses epsilon threshold to stop when
    continuation probability is negligible.
    """

    def __init__(
        self,
        condition: SoftCondition,
        body: Union[SoftPrimitive, List[SoftPrimitive]],
        max_iter: int = 64,
        epsilon: float = 1e-5,
        beta: float = 10.0,
        hard: bool = False,
    ):
        super().__init__(beta=beta, hard=hard)
        self.condition = condition
        self.body = body if isinstance(body, list) else [body]
        self.max_iter = max_iter
        self.epsilon = epsilon

        # Register child modules
        for i, mod in enumerate(self.body):
            if isinstance(mod, SoftPrimitive):
                self.add_module(f"body_{i}", mod)
        if isinstance(condition, SoftPrimitive):
            self.add_module("condition_mod", condition)

    def forward(self, state: ProgramState) -> ProgramState:
        p_after = self.condition(state)
        p_before = torch.ones_like(p_after)

        accumulate_state = state.clone()
        accumulate_state.reset()
        accumulate_state.add(state, p_before - p_after)

        i = 0
        while p_after.max() > self.epsilon and i < self.max_iter:
            for mod in self.body:
                state = mod(state)

            p_before = p_after
            p_after = p_after * self.condition(state)

            accumulate_state.add(state, p_before - p_after)
            i += 1

        accumulate_state.add(state, p_after)
        return accumulate_state

    def to_code(self, indent: int = 0) -> str:
        lines = []
        cond_code = self.condition.to_code()
        lines.append(self._indent(f"while {cond_code}:", indent))
        for mod in self.body:
            lines.append(mod.to_code(indent + 1))
        return "\n".join(lines)


class SoftFor(SoftPrimitive):
    """Standard for loop (no relaxation needed for fixed ranges).

    Ported from AlgoVision For (control_structures.py:123-186).
    """

    def __init__(
        self,
        var: str,
        range_val: int,
        body: Union[SoftPrimitive, List[SoftPrimitive]],
        beta: float = 10.0,
        hard: bool = False,
    ):
        super().__init__(beta=beta, hard=hard)
        self.var = var
        self.range_val = range_val
        self.body = body if isinstance(body, list) else [body]

        for i, mod in enumerate(self.body):
            if isinstance(mod, SoftPrimitive):
                self.add_module(f"body_{i}", mod)

    def forward(self, state: ProgramState) -> ProgramState:
        for i in range(self.range_val):
            state[self.var] = i
            for mod in self.body:
                state = mod(state)
        # Clean up loop variable
        if self.var in state.state:
            del state.state[self.var]
        return state

    def to_code(self, indent: int = 0) -> str:
        lines = []
        lines.append(self._indent(f"for {self.var} in range({self.range_val}):", indent))
        for mod in self.body:
            lines.append(mod.to_code(indent + 1))
        return "\n".join(lines)
