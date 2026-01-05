"""RLM iteration state management for ADK integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlm_adk.metadata import RLMContextMetadata, RLMIterationMetadata


@dataclass
class RLMIteration:
    """Record of a single RLM iteration."""

    iteration_number: int
    generated_code: str
    execution_result: dict[str, Any]
    stdout: str
    stderr: str
    error: str | None = None
    llm_calls_made: int = 0
    metadata: RLMIterationMetadata | None = None

    def format_for_prompt(self) -> str:
        """Format this iteration for inclusion in LLM prompt."""
        parts = [
            f"=== Iteration {self.iteration_number} ===",
            f"Code:\n```python\n{self.generated_code}\n```",
        ]

        if self.stdout:
            parts.append(f"Output:\n{self.stdout}")

        if self.stderr:
            parts.append(f"Stderr:\n{self.stderr}")

        if self.error:
            parts.append(f"Error:\n{self.error}")

        if self.llm_calls_made > 0:
            parts.append(f"(Made {self.llm_calls_made} sub-LM calls)")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "iteration_number": self.iteration_number,
            "generated_code": self.generated_code,
            "execution_result": self.execution_result,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "llm_calls_made": self.llm_calls_made,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }


@dataclass
class RLMSessionState:
    """Tracks state across RLM iterations within a session."""

    session_id: str
    context_description: str = ""
    context_metadata: RLMContextMetadata | None = None
    iterations: list[RLMIteration] = field(default_factory=list)
    final_answer: str | None = None
    final_var_name: str | None = None
    total_llm_calls: int = 0

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)

    @property
    def iteration_history(self) -> str:
        """Format all iterations for inclusion in prompt."""
        if not self.iterations:
            return "(No previous iterations)"
        return "\n\n".join(it.format_for_prompt() for it in self.iterations)

    def add_iteration(
        self,
        generated_code: str,
        execution_result: dict[str, Any],
        metadata: RLMIterationMetadata | None = None,
    ) -> RLMIteration:
        """Add a new iteration record."""
        iteration = RLMIteration(
            iteration_number=self.iteration_count + 1,
            generated_code=generated_code,
            execution_result=execution_result,
            stdout=execution_result.get("stdout", ""),
            stderr=execution_result.get("stderr", ""),
            error=execution_result.get("error_message"),
            llm_calls_made=execution_result.get("llm_calls", 0),
            metadata=metadata,
        )
        self.iterations.append(iteration)
        self.total_llm_calls += iteration.llm_calls_made
        return iteration

    def to_dict(self) -> dict:
        """Serialize for storage in ADK session state."""
        return {
            "session_id": self.session_id,
            "context_description": self.context_description,
            "context_metadata": self.context_metadata.to_dict() if self.context_metadata else None,
            "iterations": [it.to_dict() for it in self.iterations],
            "iteration_count": self.iteration_count,
            "iteration_history": self.iteration_history,
            "final_answer": self.final_answer,
            "total_llm_calls": self.total_llm_calls,
        }


def get_or_create_rlm_state(
    session_state: dict,
    session_id: str = "default",
) -> RLMSessionState:
    """Get existing RLM state from ADK session or create new."""
    state_key = f"_rlm_state_{session_id}"

    if state_key not in session_state:
        session_state[state_key] = RLMSessionState(session_id=session_id)

    return session_state[state_key]
