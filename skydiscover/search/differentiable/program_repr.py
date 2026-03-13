"""
DifferentiableProgram: extends SkyDiscover's Program with a differentiable
primitive graph and learned parameters.

Maintains dual representation:
- structure: PrimitiveGraph (differentiable, for gradient optimization)
- solution: str (discretized code, for standard evaluation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from skydiscover.search.base_database import Program


@dataclass
class DifferentiableProgram(Program):
    """A program that can be both differentiable and discretized.

    Extends the base Program dataclass with:
    - structure: the differentiable PrimitiveGraph (None if code-only)
    - parameter_state: serialized learned parameters
    - optimization_history: loss trajectory during gradient optimization
    - beta_final: final beta value used during optimization
    - structure_hash: hash for deduplication of structures
    """

    # Differentiable-specific fields
    structure_hash: Optional[str] = None
    optimization_history: Optional[List[float]] = field(default_factory=list)
    beta_final: Optional[float] = None
    optimization_steps_used: int = 0
    mode: str = "hybrid"  # "hybrid", "gradient_pure", "gradient_enhanced"

    # Note: structure (PrimitiveGraph) is not stored in the dataclass
    # because it's a torch.nn.Module and not serializable via asdict().
    # Instead, we store structure_hash and reconstruct from the database's
    # structure archive when needed.

    @classmethod
    def from_program(cls, program: Program, **kwargs) -> DifferentiableProgram:
        """Create a DifferentiableProgram from a base Program."""
        return cls(
            id=program.id,
            solution=program.solution,
            language=program.language,
            metrics=program.metrics,
            iteration_found=program.iteration_found,
            parent_id=program.parent_id,
            other_context_ids=program.other_context_ids,
            parent_info=program.parent_info,
            context_info=program.context_info,
            timestamp=program.timestamp,
            metadata=program.metadata,
            artifacts=program.artifacts,
            prompts=program.prompts,
            generation=program.generation,
            **kwargs,
        )
