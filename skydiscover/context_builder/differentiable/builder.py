"""
Context builder for differentiable search mode.

Prompts the LLM to compose algorithms from the available differentiable
primitives library rather than writing arbitrary code. The output is
parsed into a PrimitiveGraph.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from skydiscover.config import Config
from skydiscover.context_builder.base import ContextBuilder
from skydiscover.context_builder.utils import TemplateManager
from skydiscover.search.base_database import Program

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = str(Path(__file__).parent / "templates")

# Available primitives description for the LLM
PRIMITIVES_REFERENCE = """
## Available Differentiable Primitives

### Conditions (return soft probability in [0,1])
- SoftGT(left, right): Greater-than comparison. sigmoid(beta * (left - right))
- SoftLT(left, right): Less-than comparison. sigmoid(-beta * (left - right))
- SoftEq(left, right): Equality check. 1/cosh(beta/2 * (left - right))^2
- SoftNEq(left, right): Not-equal check.

### Control Flow
- SoftIf(condition, if_true, if_false): Executes both branches, merges probabilistically.
- SoftWhile(condition, body, max_iter): Loop with probabilistic termination.
- SoftFor(var, range_val, body): Standard loop over fixed range.

### Functions
- SoftMin(var_a, var_b, result_var): Differentiable minimum.
- SoftMax(var_a, var_b, result_var): Differentiable maximum.
- SoftSwap(var_a, var_b, condition): Conditional swap of two variables.
- SoftSelect(options, result_var): Learnable selection over alternatives.

### Composition
- Sequence(primitives): Sequential composition.
- WeightedChoice(options): Learnable architecture choice (gradient-optimized).
- PrimitiveGraph(inputs, outputs, variables, operations): Complete algorithm.
- LetAssign(target, source): Variable assignment.

### Key Concepts
- All primitives use inverse temperature `beta` for relaxation sharpness
- During search: soft mode (differentiable, continuous probabilities)
- During evaluation: hard mode (discrete decisions, executable code)
- WeightedChoice weights are optimized by gradient descent
"""


class DifferentiableContextBuilder(ContextBuilder):
    """Builds prompts for LLM to propose algorithm structures using primitives.

    Instead of asking for arbitrary code, constrains the LLM to compose
    from the available differentiable primitives library.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.template_manager = TemplateManager(_TEMPLATES_DIR, self.context_config.template_dir)

    def build_prompt(
        self,
        current_program: Union[Program, Dict[str, Program]],
        context: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """Build a prompt for structure proposal.

        Args:
            current_program: Current best program to improve upon.
            context: Dict with program_metrics, other_context_programs, etc.

        Returns:
            Dict with "system" and "user" keys.
        """
        context = context or {}

        # Unwrap dict-wrapped programs
        if isinstance(current_program, dict):
            info_key = list(current_program.keys())[0]
            program = current_program[info_key]
        else:
            program = current_program

        # Build system message
        system_message = self._build_system_message()

        # Build user message
        user_message = self._build_user_message(program, context)

        return {"system": system_message, "user": user_message}

    def _build_system_message(self) -> str:
        """Build the system message with primitives reference."""
        # Try to load from template
        try:
            template = self.template_manager.get_template("structure_proposal_system_message")
            if template:
                return template
        except Exception:
            pass

        # Fallback: inline system message
        custom_system = self.context_config.system_message or ""
        return f"""{custom_system}

You are an algorithm designer that composes algorithms from differentiable primitives.
Your goal is to propose algorithm structures that can be optimized via gradient descent.

{PRIMITIVES_REFERENCE}

When proposing an algorithm, output a Python code block that constructs a PrimitiveGraph
using these primitives. The graph should be a valid composition that can be gradient-optimized.

Example structure proposal:
```python
graph = PrimitiveGraph(
    inputs=["array"],
    outputs=["sorted_array"],
    variables={{"temp": 0.0, "swapped": 1.0}},
    operations=[
        SoftFor("i", n,
            SoftFor("j", n-1, [
                SoftSwap("array[j]", "array[j+1]",
                    condition=SoftGT("array[j]", "array[j+1]"))
            ])
        )
    ]
)
```
"""

    def _build_user_message(self, program: Program, context: Dict[str, Any]) -> str:
        """Build user message with current state and improvement request."""
        parts = []

        # Current best score
        metrics = context.get("program_metrics", program.metrics)
        if metrics:
            score = metrics.get("combined_score", "N/A")
            parts.append(f"## Current Best Score: {score}")
            other_metrics = {k: v for k, v in metrics.items() if k != "combined_score"}
            if other_metrics:
                parts.append("### Detailed Metrics")
                for k, v in other_metrics.items():
                    parts.append(f"- {k}: {v}")

        # Current solution
        parts.append("\n## Current Best Solution")
        parts.append(f"```python\n{program.solution}\n```")

        # Context programs
        other_programs = context.get("other_context_programs", [])
        if other_programs:
            parts.append("\n## Other High-Performing Solutions")
            for i, p in enumerate(other_programs[:3]):
                score = p.metrics.get("combined_score", "N/A") if p.metrics else "N/A"
                parts.append(f"\n### Solution {i+1} (score={score})")
                parts.append(f"```python\n{p.solution[:500]}\n```")

        # Evaluator feedback
        if program.artifacts:
            parts.append("\n## Evaluator Feedback")
            for k, v in program.artifacts.items():
                if isinstance(v, str):
                    parts.append(f"- {k}: {v}")

        # Task instruction
        parts.append(
            "\n## Task\n"
            "Propose a new or improved algorithm structure using the differentiable primitives. "
            "Focus on the algorithmic structure - continuous parameters will be optimized "
            "automatically via gradient descent. Output a Python code block constructing "
            "a PrimitiveGraph."
        )

        return "\n".join(parts)
