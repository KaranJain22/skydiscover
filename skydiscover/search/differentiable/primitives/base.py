"""
Base classes for differentiable primitives.

Adapted from AlgoVision's core.py (State, AlgoModule) for SkyDiscover's
discovery context. Key additions: to_code() for generating executable
Python from discretized primitives.
"""

from __future__ import annotations

import copy
from abc import abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class ProgramState:
    """Mutable state container for differentiable algorithm execution.

    Adapted from AlgoVision's State class. Supports:
    - Named tensor variables with batch dimension
    - Probabilistic merging for soft control flow
    - Weighted accumulation for soft while loops
    - Clone/reset for branching execution
    """

    def __init__(self, batch_size: int, device: Optional[torch.device] = None):
        self.state: Dict[str, Any] = {}
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")

    def __getitem__(self, key: str) -> Any:
        return self.state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.state[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.state

    def keys(self):
        return self.state.keys()

    def clone(self) -> ProgramState:
        """Deep clone of the state for branching."""
        new_state = ProgramState(self.batch_size, self.device)
        for key, value in self.state.items():
            if isinstance(value, torch.Tensor):
                new_state.state[key] = value.clone()
            elif isinstance(value, (int, float)):
                new_state.state[key] = value
            else:
                new_state.state[key] = copy.deepcopy(value)
        return new_state

    def merge(self, other: ProgramState, p: torch.Tensor) -> None:
        """Probabilistically merge another state into this one.

        For each variable: self[key] = self[key] * (1-p) + other[key] * p

        Adapted from AlgoVision's State.merge (core.py:203-247).
        """
        for key, value in other.state.items():
            if isinstance(value, (int, float)):
                # Hard integer/float values must match
                continue
            if not isinstance(value, torch.Tensor):
                continue

            p_0 = 1 - p
            p_1 = p
            while len(p_0.shape) < len(value.shape):
                p_0 = p_0.unsqueeze(-1)
                p_1 = p_1.unsqueeze(-1)

            self.state[key] = self.state[key] * p_0 + value * p_1

    def add(self, other: ProgramState, p: torch.Tensor) -> None:
        """Weighted addition of another state (for while loop accumulation).

        self[key] += other[key] * p

        Adapted from AlgoVision's State.add (core.py:272-299).
        """
        for key, value in other.state.items():
            if isinstance(value, (int, float)):
                self.state[key] = value
                continue
            if not isinstance(value, torch.Tensor):
                continue

            p_0 = p
            while len(p_0.shape) < len(value.shape):
                p_0 = p_0.unsqueeze(-1)

            self.state[key] = self.state[key] + p_0 * value

    def probabilistic_update(self, key: str, value: Any, p: torch.Tensor) -> None:
        """Update a single variable probabilistically.

        self[key] = p * value + (1-p) * self[key]
        """
        if not isinstance(value, torch.Tensor):
            if isinstance(value, (int, float)):
                value = torch.tensor(value, dtype=self.state[key].dtype, device=self.device)
            else:
                return

        while len(value.shape) < len(self.state[key].shape):
            value = value.unsqueeze(-1)

        self.state[key] = p * value + (1 - p) * self.state[key]

    def reset(self) -> None:
        """Reset all tensor variables to zero."""
        for key in self.state:
            if isinstance(self.state[key], torch.Tensor):
                self.state[key] = torch.zeros_like(self.state[key])

    def get_device(self) -> torch.device:
        for value in self.state.values():
            if isinstance(value, torch.Tensor):
                return value.device
        return self.device


class SoftPrimitive(nn.Module):
    """Base class for all differentiable algorithm primitives.

    Adapted from AlgoVision's AlgoModule. Each primitive:
    - Has a beta (inverse temperature) controlling relaxation sharpness
    - Supports hard mode (discrete) vs soft mode (differentiable)
    - Can generate executable code via to_code()
    """

    def __init__(self, beta: float = 10.0, hard: bool = False):
        super().__init__()
        self._beta = beta
        self.hard = hard

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        self._beta = value

    def set_beta_recursive(self, beta: float) -> None:
        """Set beta on this primitive and all children."""
        self.beta = beta
        for child in self.children():
            if isinstance(child, SoftPrimitive):
                child.set_beta_recursive(beta)

    def set_hard_recursive(self, hard: bool) -> None:
        """Set hard mode on this primitive and all children."""
        self.hard = hard
        for child in self.children():
            if isinstance(child, SoftPrimitive):
                child.set_hard_recursive(hard)

    @abstractmethod
    def forward(self, state: ProgramState) -> ProgramState:
        """Execute the primitive on a program state."""
        ...

    @abstractmethod
    def to_code(self, indent: int = 0) -> str:
        """Generate executable Python code for the discretized primitive."""
        ...

    def _indent(self, code: str, level: int) -> str:
        """Indent code by the given number of levels (4 spaces each)."""
        prefix = "    " * level
        return "\n".join(prefix + line if line.strip() else line for line in code.split("\n"))
