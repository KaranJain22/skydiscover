"""
GradientEnhancedDatabase: standard program database with optional
tracking of gradient refinement history.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


class GradientEnhancedDatabase(ProgramDatabase):
    """Database for gradient-enhanced evolution.

    Same as TopK but tracks which programs were gradient-refined.
    """

    def __init__(self, name: str, config: DatabaseConfig, **kwargs):
        super().__init__(name, config, **kwargs)
        self.initial_program = None
        self.refined_program_ids: set = set()

    def add(self, program: Program, iteration: Optional[int] = None, **kwargs) -> str:
        """Add a program to the database."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        # Track if this program was gradient-refined
        if kwargs.get("gradient_refined", False):
            self.refined_program_ids.add(program.id)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        logger.debug(f"Added program {program.id} to gradient_enhanced database")
        return program.id

    def get_refinement_stats(self) -> Dict[str, Any]:
        """Get statistics about gradient refinement."""
        total = len(self.programs)
        refined = len(self.refined_program_ids)
        return {
            "total_programs": total,
            "refined_programs": refined,
            "refinement_ratio": refined / max(total, 1),
        }

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Program, List[Program]]:
        """Sample parent and context programs (Top-K strategy)."""
        if not self.programs:
            raise ValueError("Cannot sample: no programs in database")

        total_needed = (num_context_programs or 4) + 1
        top_programs = self.get_top_programs(total_needed)

        if not top_programs:
            raise ValueError("Cannot sample: no programs available")

        if len(top_programs) < 2:
            return top_programs[0], [top_programs[0]]

        return top_programs[0], top_programs[1 : num_context_programs + 1]
