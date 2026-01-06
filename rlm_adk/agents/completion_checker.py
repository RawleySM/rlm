"""Completion checker agent for RLM iteration loop.

Uses BaseAgent to detect FINAL/FINAL_VAR patterns and signal
loop termination via escalation.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from rlm_adk.rlm_repl import find_final_answer
from rlm_adk.rlm_state import get_or_create_rlm_state


class RLMCompletionChecker(BaseAgent):
    """Checks for FINAL/FINAL_VAR patterns and signals loop termination.

    This is a BaseAgent (not LlmAgent) because it needs to:
    1. Perform deterministic checks (no LLM needed)
    2. Yield EventActions(escalate=True) to exit the LoopAgent
    """

    def __init__(self, name: str = "rlm_completion_checker"):
        super().__init__(
            name=name,
            description="Checks if RLM iteration has produced a FINAL answer",
        )

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        """Check for completion and optionally escalate to exit loop."""

        session_state = ctx.session.state
        generated_code = session_state.get("generated_code", "")
        execution_result = session_state.get("execution_result", {})

        session_id = session_state.get("rlm_session_id", "default")
        rlm_state = get_or_create_rlm_state(session_state, session_id)

        # Check for FINAL pattern in generated code
        final_answer = find_final_answer(generated_code)

        # Also check stdout from execution
        if not final_answer:
            stdout = execution_result.get("stdout", "")
            final_answer = find_final_answer(stdout)

        iteration = rlm_state.iteration_count

        if final_answer:
            # Handle FINAL_VAR pattern
            if final_answer.startswith("__FINAL_VAR__:"):
                var_name = final_answer.split(":", 1)[1]
                rlm_state.final_var_name = var_name
                rlm_state.final_answer = f"[Result stored in variable: {var_name}]"
            else:
                rlm_state.final_answer = final_answer

            # Store final answer in session state
            session_state["rlm_final_answer"] = rlm_state.final_answer
            if rlm_state.final_var_name:
                session_state["rlm_final_var_name"] = rlm_state.final_var_name
            session_state["rlm_iteration_count"] = iteration
            session_state["rlm_total_llm_calls"] = rlm_state.total_llm_calls
            session_state["rlm_termination_reason"] = "FINAL"

            # Signal loop termination via escalation
            yield Event(
                author=self.name,
                actions=EventActions(escalate=True),
            )

        else:
            error = execution_result.get("error_message")
            if error:
                # Continue loop - let code_generator see the error
                yield Event(author=self.name)
            else:
                # Normal iteration complete
                yield Event(author=self.name)
