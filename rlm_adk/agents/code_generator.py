"""Code generator agent for RLM iteration loop.

This agent generates Python code that uses llm_query() for recursive
decomposition of problems. Uses the composed system prompt from
rlm/utils/prompts.py with healthcare extensions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from google.adk.agents import LlmAgent
from rlm_adk.callbacks import get_code_generator_callbacks
from rlm_adk.prompts import get_code_generator_instruction


def make_code_generator() -> LlmAgent:
    """Create the code generator agent.

    Returns:
        LlmAgent that generates Python code with llm_query() calls.
    """
    # Get the composed instruction (RLM_SYSTEM_PROMPT + healthcare extension)
    # Use ADK state placeholders so the prompt reflects live session data
    instruction = get_code_generator_instruction(
        context_description="{context_description}",
        iteration_history="{iteration_history}",
        user_query="{user_query}",
    )

    # Get callback bundle for state/error management
    callbacks = get_code_generator_callbacks()

    return LlmAgent(
        name="rlm_code_generator",
        model="gemini-3-pro",
        description="Generates Python code for recursive problem decomposition using llm_query()",
        instruction=instruction,
        output_key="generated_code",
        # Attach callbacks for state management
        before_agent_callback=callbacks.get("before_agent_callback"),
        before_model_callback=callbacks.get("before_model_callback"),
        after_model_callback=callbacks.get("after_model_callback"),
        on_model_error_callback=callbacks.get("on_model_error_callback"),
    )
