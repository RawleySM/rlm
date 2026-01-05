"""Context setup agent for RLM workflow.

This agent loads data into the RLM context variable using the available
tools, preparing data for iterative processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool


def make_context_setup_agent() -> LlmAgent:
    """Create the context setup agent.

    Returns:
        LlmAgent that loads data into RLM context.
    """
    from rlm_adk.tools.rlm_tools import rlm_load_context

    return LlmAgent(
        name="rlm_context_loader",
        model="gemini-3-pro",
        description="Loads data into RLM context for iterative processing",
        instruction="""You are the context setup component of the RLM system.

Your task is to load data into the RLM context variable using the rlm_load_context tool.

The user's query will describe what data needs to be loaded. Use the tool to:
1. Load data from Unity Catalog volumes or other sources
2. Store it in the 'context' variable for use in RLM iterations
3. Provide a description of the loaded data

If the query already includes context or doesn't need context loading, you can skip
the tool call and just set a context_description explaining what's available.

After loading, store a clear description in the output key 'context_setup_result' that includes:
- What data was loaded
- Size/structure of the data
- Any relevant metadata

This description will be passed to the code generator to help it understand what's available.
""",
        tools=[FunctionTool(rlm_load_context)],
        output_key="context_setup_result",
    )
