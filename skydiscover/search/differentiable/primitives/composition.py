"""
Composition primitives for building algorithm structures.

These are new to SkyDiscover (not in AlgoVision, which hardcodes structures).
They enable the LLM to propose and gradient descent to optimize algorithm
architectures as DAGs of differentiable primitives.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from skydiscover.search.differentiable.primitives.base import (
    ProgramState,
    SoftPrimitive,
)


class Sequence(SoftPrimitive):
    """Sequential composition of primitives."""

    def __init__(
        self,
        primitives: List[SoftPrimitive],
        beta: float = 10.0,
        hard: bool = False,
    ):
        super().__init__(beta=beta, hard=hard)
        self.primitives = nn.ModuleList(primitives)

    def forward(self, state: ProgramState) -> ProgramState:
        for primitive in self.primitives:
            state = primitive(state)
        return state

    def to_code(self, indent: int = 0) -> str:
        lines = []
        for primitive in self.primitives:
            lines.append(primitive.to_code(indent))
        return "\n".join(lines)


class LetAssign(SoftPrimitive):
    """Variable assignment primitive.

    Assigns the value of one variable (or a constant) to another.
    """

    def __init__(
        self,
        target: str,
        source: str,
        beta: float = 10.0,
        hard: bool = False,
    ):
        super().__init__(beta=beta, hard=hard)
        self.target = target
        self.source = source

    def forward(self, state: ProgramState) -> ProgramState:
        if self.source in state:
            state[self.target] = state[self.source].clone()
        return state

    def to_code(self, indent: int = 0) -> str:
        return self._indent(f"{self.target} = {self.source}", indent)


class WeightedChoice(SoftPrimitive):
    """Learnable selection over alternative primitive sub-graphs.

    Weights are nn.Parameters. During search, all options are evaluated
    and results are blended by softmax(beta * weights). During evaluation,
    argmax(weights) selects one option.
    """

    def __init__(
        self,
        options: List[SoftPrimitive],
        beta: float = 10.0,
        hard: bool = False,
    ):
        super().__init__(beta=beta, hard=hard)
        self.options = nn.ModuleList(options)
        self.arch_weights = nn.Parameter(torch.zeros(len(options)))

    def forward(self, state: ProgramState) -> ProgramState:
        if self.hard:
            idx = self.arch_weights.argmax().item()
            return self.options[idx](state)

        probs = torch.softmax(self.beta * self.arch_weights, dim=0)

        # Execute all options and blend results
        option_states = []
        for option in self.options:
            s = state.clone()
            s = option(s)
            option_states.append(s)

        # Weighted merge into first state
        result = option_states[0]
        for key in list(result.keys()):
            val = result[key]
            if isinstance(val, torch.Tensor):
                blended = probs[0] * val
                for i in range(1, len(option_states)):
                    other = option_states[i][key]
                    if isinstance(other, torch.Tensor):
                        blended = blended + probs[i] * other
                result[key] = blended

        return result

    def to_code(self, indent: int = 0) -> str:
        idx = self.arch_weights.argmax().item()
        return self.options[idx].to_code(indent)

    def selected_index(self) -> int:
        return self.arch_weights.argmax().item()


class PrimitiveGraph(SoftPrimitive):
    """A directed acyclic graph of primitives representing an algorithm.

    This is the top-level container for a differentiable algorithm.
    It manages inputs, outputs, variables, and a sequence of operations.
    """

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        variables: Optional[Dict[str, Any]] = None,
        operations: Optional[List[SoftPrimitive]] = None,
        beta: float = 10.0,
        hard: bool = False,
    ):
        super().__init__(beta=beta, hard=hard)
        self.input_names = inputs
        self.output_names = outputs
        self.variable_defaults = variables or {}
        self.operations = nn.ModuleList(operations or [])

    def forward(self, state: ProgramState) -> ProgramState:
        # Initialize variables with defaults
        for name, default_val in self.variable_defaults.items():
            if name not in state:
                if isinstance(default_val, torch.Tensor):
                    state[name] = default_val.clone().expand(state.batch_size, *default_val.shape)
                else:
                    state[name] = torch.tensor(
                        default_val, dtype=torch.float32, device=state.device
                    ).expand(state.batch_size)

        # Execute operations in sequence
        for op in self.operations:
            state = op(state)

        return state

    def to_code(self, indent: int = 0) -> str:
        lines = []

        # Function signature
        args = ", ".join(self.input_names)
        lines.append(self._indent(f"def algorithm({args}):", indent))

        # Variable initialization
        for name, default_val in self.variable_defaults.items():
            if isinstance(default_val, torch.Tensor):
                val_str = repr(default_val.tolist())
            else:
                val_str = repr(default_val)
            lines.append(self._indent(f"{name} = {val_str}", indent + 1))

        # Operations
        for op in self.operations:
            lines.append(op.to_code(indent + 1))

        # Return
        if len(self.output_names) == 1:
            lines.append(self._indent(f"return {self.output_names[0]}", indent + 1))
        else:
            outputs = ", ".join(self.output_names)
            lines.append(self._indent(f"return {outputs}", indent + 1))

        return "\n".join(lines)

    def structure_hash(self) -> str:
        """Hash the structure (ignoring learned parameter values) for dedup."""
        desc = self._describe_structure()
        return hashlib.md5(json.dumps(desc, sort_keys=True).encode()).hexdigest()

    def _describe_structure(self) -> Dict[str, Any]:
        """Describe the graph structure without parameter values."""
        return {
            "inputs": self.input_names,
            "outputs": self.output_names,
            "variables": list(self.variable_defaults.keys()),
            "operations": [type(op).__name__ for op in self.operations],
            "num_params": sum(p.numel() for p in self.parameters()),
        }
