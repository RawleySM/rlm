"""Result formatter agent for RLM workflow.

Formats the final RLM answer with execution metrics for user presentation.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent


def make_result_formatter() -> LlmAgent:
    """Create the result formatter agent.

    Returns:
        LlmAgent that formats RLM final results.
    """

    return LlmAgent(
        name="rlm_result_formatter",
        model="gemini-3-pro",
        description="Formats RLM final results with metrics for user presentation",
        instruction="""You are the result formatting component of the RLM system.

Your task is to format the final RLM answer into a clear, user-friendly response.

You have access to the following state variables:
- {rlm_final_answer}: The final answer from the RLM iteration loop
- {rlm_iteration_count}: Number of iterations performed
- {rlm_total_llm_calls}: Total sub-LM calls made during processing
- {rlm_execution_metrics}: Detailed execution metrics

Format the response as follows:

# Result

[The final answer from rlm_final_answer, presented clearly]

## Processing Summary

- Iterations: {rlm_iteration_count}
- Sub-LM Calls: {rlm_total_llm_calls}
- Total Time: {from metrics}

[If available, include any relevant statistics or insights from the execution metrics]

Keep the formatting professional and easy to read. Focus on the user's answer first,
with metrics as supporting information.

If rlm_final_answer indicates that a variable was stored (e.g., "[Result stored in variable: result]"),
explain that the result is available in the REPL context and can be accessed via that variable name.
""",
        output_key="rlm_formatted_result",
    )
