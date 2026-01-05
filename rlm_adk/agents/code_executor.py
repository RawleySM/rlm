"""Code executor agent for RLM iteration loop.

Extracts code blocks from generated_code and executes them in the
RLM REPL environment with real llm_query() access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool, ToolContext

from rlm_adk.callbacks import get_code_executor_callbacks
from rlm_adk.metadata import RLMExecutionMetadata


def execute_rlm_iteration(tool_context: ToolContext) -> dict[str, Any]:
    """Execute the generated code from the code_generator.

    This tool:
    1. Extracts code blocks from {generated_code}
    2. Executes them in the RLM REPL with llm_query() access
    3. Updates iteration_history for the next iteration
    4. Returns execution results with custom_metadata

    Returns:
        dict with status, stdout, stderr, error_message, llm_calls, custom_metadata
    """
    import time

    from rlm_adk.llm_bridge import (
        create_llm_query_bridge,
        create_llm_query_batched_bridge,
        reset_sub_lm_call_metadata,
        get_sub_lm_call_metadata,
    )
    from rlm_adk.rlm_repl import find_code_blocks, get_or_create_repl_session
    from rlm_adk.rlm_state import get_or_create_rlm_state

    start_time = time.time()

    # Reset sub-LM call tracking for this iteration
    reset_sub_lm_call_metadata()

    # Get generated code from previous agent
    generated_code = tool_context.state.get("generated_code", "")

    if not generated_code:
        return {
            "status": "error",
            "error_message": "No generated_code found in state",
            "stdout": "",
            "stderr": "",
            "llm_calls": 0,
            "custom_metadata": RLMExecutionMetadata(
                iteration_number=0,
                status="error",
                error_type="missing_code",
            ).to_dict(),
        }

    # Extract code blocks
    code_blocks = find_code_blocks(generated_code)

    if not code_blocks:
        return {
            "status": "no_code",
            "message": "No code blocks found in generated response",
            "raw_response": generated_code[:500],
            "stdout": "",
            "stderr": "",
            "llm_calls": 0,
            "custom_metadata": RLMExecutionMetadata(
                iteration_number=0,
                status="no_code",
            ).to_dict(),
        }

    # Get RLM state
    session_id = tool_context.state.get("rlm_session_id", "default")
    rlm_state = get_or_create_rlm_state(tool_context.state, session_id)

    # Create REAL llm_query bridge
    invocation_ctx = getattr(tool_context, "invocation_context", None)

    llm_query_fn = create_llm_query_bridge(invocation_ctx)
    llm_query_batched_fn = create_llm_query_batched_bridge(invocation_ctx)

    # Get or create REPL session
    repl = get_or_create_repl_session(
        session_id=session_id,
        llm_query_fn=llm_query_fn,
        llm_query_batched_fn=llm_query_batched_fn,
        context=tool_context.state.get("rlm_context"),
    )

    # Execute each code block
    all_stdout = []
    all_stderr = []
    total_llm_calls = 0
    last_status = "success"
    error_message = None
    variables_created = []

    for i, code in enumerate(code_blocks):
        result = repl.execute_code(code)

        all_stdout.append(result.get("stdout", ""))
        all_stderr.append(result.get("stderr", ""))
        total_llm_calls += result.get("llm_calls", 0)

        # Track new variables (if REPL supports it)
        if "variables" in result:
            variables_created.extend(result["variables"])

        if result.get("status") == "error":
            last_status = "error"
            error_message = f"Block {i+1}: {result.get('error_message', 'Unknown error')}"

    # Calculate execution time
    execution_time_ms = (time.time() - start_time) * 1000

    # Build execution metadata
    exec_metadata = RLMExecutionMetadata(
        iteration_number=rlm_state.iteration_count + 1,
        blocks_executed=len(code_blocks),
        llm_calls_made=total_llm_calls,
        execution_time_ms=round(execution_time_ms, 1),
        status=last_status,
        error_type=error_message.split(":")[0] if error_message else None,
        variables_created=variables_created,
    )

    # Build execution result
    execution_result = {
        "status": last_status,
        "stdout": "\n".join(filter(None, all_stdout)),
        "stderr": "\n".join(filter(None, all_stderr)),
        "error_message": error_message,
        "llm_calls": total_llm_calls,
        "blocks_executed": len(code_blocks),
        "iteration": rlm_state.iteration_count + 1,
        "custom_metadata": exec_metadata.to_dict(),
        "sub_lm_call_details": get_sub_lm_call_metadata(),
    }

    # Update iteration history (CRITICAL for feedback loop)
    # Get iteration metadata from state if available
    iteration_metadata = tool_context.state.get("_last_iteration_metadata")

    rlm_state.add_iteration(
        generated_code=generated_code,
        execution_result=execution_result,
        metadata=iteration_metadata,
    )

    # Store formatted history in state for next code_generator call
    tool_context.state["iteration_history"] = rlm_state.iteration_history
    tool_context.state["execution_result"] = execution_result
    tool_context.state["_rlm_current_iteration"] = rlm_state.iteration_count

    # Sync REPL context back to state
    tool_context.state["rlm_context"] = repl.context

    return execution_result


execute_rlm_iteration_tool = FunctionTool(execute_rlm_iteration)


def make_code_executor() -> LlmAgent:
    """Create the code executor agent.

    Returns:
        LlmAgent that executes RLM code blocks.
    """
    callbacks = get_code_executor_callbacks()

    return LlmAgent(
        name="rlm_code_executor",
        model="gemini-3-pro",
        description="Executes Python code in RLM REPL with llm_query() access",
        instruction="""You are the code execution component of the RLM system.

Your ONLY task is to call the execute_rlm_iteration tool to run the generated code.

Call the tool immediately without any additional analysis. The tool will:
1. Extract code blocks from the generated code
2. Execute them in the REPL environment
3. Return the execution results with detailed metadata

 After execution, briefly report the results (success/error, any output).
""",
        tools=[execute_rlm_iteration_tool],
        output_key="execution_result",
        before_tool_callback=callbacks.get("before_tool_callback"),
        after_tool_callback=callbacks.get("after_tool_callback"),
    )
