"""
GradientPureDatabase: stores programs from pure gradient-based search.

Similar to DifferentiableDatabase but tracks a population of PrimitiveGraphs
that are optimized without LLM involvement.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase
from skydiscover.search.differentiable.program_repr import DifferentiableProgram

logger = logging.getLogger(__name__)


class GradientPureDatabase(ProgramDatabase):
    """Database for pure gradient search.

    Maintains a population of PrimitiveGraphs optimized via gradient
    descent. No LLM involvement in the inner loop.
    """

    def __init__(self, name: str, config: DatabaseConfig, **kwargs):
        super().__init__(name, config, **kwargs)
        self.initial_program = None
        # graph_id -> PrimitiveGraph (torch module, kept in memory)
        self.graph_population: Dict[str, Any] = {}
        self.population_scores: Dict[str, float] = {}

    def add(self, program: Program, iteration: Optional[int] = None, **kwargs) -> str:
        """Add a program to the database."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        logger.debug(f"Added program {program.id} to gradient_pure database")
        return program.id

    def register_graph(self, graph_id: str, graph: Any, score: float = 0.0) -> None:
        """Register a PrimitiveGraph in the population."""
        self.graph_population[graph_id] = graph
        self.population_scores[graph_id] = score

    def get_best_graph(self) -> Optional[Tuple[str, Any]]:
        """Get the graph with the highest score."""
        if not self.population_scores:
            return None
        best_id = max(self.population_scores, key=self.population_scores.get)
        return best_id, self.graph_population[best_id]

    def get_random_graph(self) -> Optional[Tuple[str, Any]]:
        """Get a random graph from the population."""
        import random

        if not self.graph_population:
            return None
        gid = random.choice(list(self.graph_population.keys()))
        return gid, self.graph_population[gid]

    def update_score(self, graph_id: str, score: float) -> None:
        """Update the score of a graph."""
        if graph_id in self.population_scores:
            self.population_scores[graph_id] = max(self.population_scores[graph_id], score)

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Program, List[Program]]:
        """Sample parent and context programs."""
        if not self.programs:
            raise ValueError("Cannot sample: no programs in database")

        total_needed = (num_context_programs or 4) + 1
        top_programs = self.get_top_programs(total_needed)

        if not top_programs:
            raise ValueError("Cannot sample: no programs available")

        if len(top_programs) < 2:
            return top_programs[0], [top_programs[0]]

        return top_programs[0], top_programs[1 : num_context_programs + 1]
