"""
DifferentiableDatabase: stores programs with their differentiable structures.

Maintains a structure archive of distinct algorithm architectures alongside
the standard program population. Supports sampling programs for either
structural proposal or parameter reoptimization.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase
from skydiscover.search.differentiable.program_repr import DifferentiableProgram

logger = logging.getLogger(__name__)


class DifferentiableDatabase(ProgramDatabase):
    """Database for differentiable search with structure tracking.

    Extends ProgramDatabase with:
    - Structure archive: maps structure_hash -> best program with that structure
    - Sampling for reoptimization vs new structure proposal
    """

    def __init__(self, name: str, config: DatabaseConfig, **kwargs):
        super().__init__(name, config, **kwargs)
        self.initial_program = None
        # structure_hash -> (best_score, program_id, structure_graph)
        self.structure_archive: Dict[str, Tuple[float, str, Any]] = {}
        # Keep track of which structures we've tried
        self.structure_history: List[str] = []

    def add(self, program: Program, iteration: Optional[int] = None, **kwargs) -> str:
        """Add a program to the database."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        # Track structure if it's a DifferentiableProgram
        if isinstance(program, DifferentiableProgram) and program.structure_hash:
            self._update_structure_archive(program)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        logger.debug(f"Added program {program.id} to differentiable database")
        return program.id

    def _update_structure_archive(self, program: DifferentiableProgram) -> None:
        """Track the best score achieved by each unique structure."""
        from skydiscover.utils.metrics import get_score

        score = get_score(program.metrics)
        h = program.structure_hash

        if h not in self.structure_archive:
            self.structure_archive[h] = (score, program.id, None)
            self.structure_history.append(h)
            logger.debug(f"New structure discovered: {h[:8]}... (score={score:.4f})")
        else:
            best_score, _, graph = self.structure_archive[h]
            if score > best_score:
                self.structure_archive[h] = (score, program.id, graph)

    def register_structure_graph(self, structure_hash: str, graph: Any) -> None:
        """Store a reference to the actual PrimitiveGraph for a structure hash.

        This is called by the controller after creating the graph, so the
        database can provide it for reoptimization later.
        """
        if structure_hash in self.structure_archive:
            score, pid, _ = self.structure_archive[structure_hash]
            self.structure_archive[structure_hash] = (score, pid, graph)
        else:
            self.structure_archive[structure_hash] = (0.0, "", graph)

    def get_structure_for_reoptimization(self) -> Optional[Tuple[str, Any]]:
        """Get the best structure with its graph for reoptimization.

        Returns:
            (structure_hash, graph) or None if no structures with graphs available.
        """
        best_score = float("-inf")
        best_hash = None
        best_graph = None

        for h, (score, _, graph) in self.structure_archive.items():
            if graph is not None and score > best_score:
                best_score = score
                best_hash = h
                best_graph = graph

        if best_hash is None:
            return None

        return best_hash, best_graph

    def get_num_unique_structures(self) -> int:
        """Return the number of unique structures discovered."""
        return len(self.structure_archive)

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Program, List[Program]]:
        """Sample parent and context programs for the next iteration.

        Uses top-K strategy: best program as parent, next K as context.
        """
        if not self.programs:
            raise ValueError("Cannot sample: no programs in database")

        total_needed = (num_context_programs or 4) + 1
        top_programs = self.get_top_programs(total_needed)

        if not top_programs:
            raise ValueError("Cannot sample: no programs available")

        if len(top_programs) < 2:
            parent = top_programs[0]
            context_programs = [top_programs[0]]
        else:
            parent = top_programs[0]
            context_programs = top_programs[1 : num_context_programs + 1]

        return parent, context_programs
