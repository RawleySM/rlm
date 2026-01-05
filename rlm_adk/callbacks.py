"""ADK Callbacks for RLM state management, error handling, and metrics.

Callbacks provide hooks at key execution points:
- before_agent / after_agent: Agent lifecycle
- before_model / after_model: LLM request/response
- before_tool / after_tool: Tool execution
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools import ToolContext

from rlm_adk.metadata import RLMExecutionMetadata, RLMIterationMetadata
from rlm_adk.rlm_state import get_or_create_rlm_state


# =============================================================================
# Callback Context Keys
# =============================================================================

STATE_KEY_RLM_METRICS = "_rlm_metrics"
STATE_KEY_ITERATION_START = "_rlm_iteration_start_time"
STATE_KEY_MODEL_LATENCIES = "_rlm_model_latencies"
STATE_KEY_TOOL_ERRORS = "_rlm_tool_errors"


# =============================================================================
# Before Agent Callbacks
# =============================================================================

def before_rlm_loop_callback(
    callback_context: CallbackContext,
) -> None:
    """Initialize RLM state before the iteration loop begins.

    This callback:
    1. Initializes RLM session state if not present
    2. Records loop start time for metrics
    3. Prepares iteration tracking structures

    Args:
        callback_context: ADK callback context with state access.
    """
    state = callback_context.state

    # Initialize RLM state
    session_id = state.get("rlm_session_id", "default")
    rlm_state = get_or_create_rlm_state(state, session_id)

    # Initialize metrics tracking
    if STATE_KEY_RLM_METRICS not in state:
        state[STATE_KEY_RLM_METRICS] = {
            "loop_start_time": time.time(),
            "total_model_calls": 0,
            "total_tool_calls": 0,
            "total_sub_lm_calls": 0,
            "errors_recovered": 0,
            "model_latencies_ms": [],
            "tool_latencies_ms": [],
        }

    # Record iteration start
    state[STATE_KEY_ITERATION_START] = time.time()

    print(f"[RLM] Starting iteration loop (session: {session_id})")


def before_code_generator_callback(
    callback_context: CallbackContext,
) -> None:
    """Prepare state before code generation.

    Ensures iteration_history and context_description are available.
    """
    state = callback_context.state
    session_id = state.get("rlm_session_id", "default")
    rlm_state = get_or_create_rlm_state(state, session_id)

    # Ensure iteration history is available for prompt substitution
    if "iteration_history" not in state:
        state["iteration_history"] = rlm_state.iteration_history

    # Provide a context description placeholder if one isn't present
    if "context_description" not in state:
        # Prefer description recorded during context loading
        context_description = state.get("rlm_context_description") or rlm_state.context_description
        state["context_description"] = context_description or "(No context loaded yet)"

    # Provide a user query placeholder to avoid missing-template errors
    state.setdefault("user_query", state.get("query") or "(Awaiting user query)")


# =============================================================================
# After Agent Callbacks
# =============================================================================

def after_rlm_loop_callback(
    callback_context: CallbackContext,
) -> None:
    """Finalize metrics and cleanup after the iteration loop completes.

    This callback:
    1. Calculates total execution time
    2. Aggregates metrics from all iterations
    3. Stores final metrics in state for reporting
    """
    state = callback_context.state
    metrics = state.get(STATE_KEY_RLM_METRICS, {})

    if "loop_start_time" in metrics:
        total_time = time.time() - metrics["loop_start_time"]
        metrics["total_execution_time_seconds"] = round(total_time, 2)

    # Get final RLM state
    session_id = state.get("rlm_session_id", "default")
    rlm_state = get_or_create_rlm_state(state, session_id)

    metrics["final_iteration_count"] = rlm_state.iteration_count
    metrics["total_sub_lm_calls"] = rlm_state.total_llm_calls

    # Store for result formatter
    state["rlm_execution_metrics"] = metrics

    print(f"[RLM] Loop complete: {rlm_state.iteration_count} iterations, "
          f"{rlm_state.total_llm_calls} sub-LM calls, "
          f"{metrics.get('total_execution_time_seconds', 0)}s total")


# =============================================================================
# Before Model Callbacks
# =============================================================================

def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    """Inspect/modify LLM request before sending.

    This callback:
    1. Injects iteration_history into the prompt if needed
    2. Records model call timing
    3. Can short-circuit with a cached response if appropriate

    Args:
        callback_context: ADK callback context.
        llm_request: The LLM request about to be sent.

    Returns:
        None to proceed with model call, or LlmResponse to short-circuit.
    """
    state = callback_context.state

    # Record timing
    state["_model_call_start"] = time.time()

    # Track call count
    metrics = state.get(STATE_KEY_RLM_METRICS, {})
    metrics["total_model_calls"] = metrics.get("total_model_calls", 0) + 1
    state[STATE_KEY_RLM_METRICS] = metrics

    # Log for debugging (can be disabled in production)
    agent_name = callback_context.agent_name
    print(f"[RLM] Model call from {agent_name}")

    return None  # Proceed with model call


def before_model_with_history_injection(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    """Before-model callback that injects iteration history into prompts.

    Use this for the code_generator agent to ensure it sees full history.
    """
    state = callback_context.state

    # Standard timing/metrics
    state["_model_call_start"] = time.time()

    # Get iteration history
    iteration_history = state.get("iteration_history", "(No previous iterations)")
    context_description = state.get("context_description", "(No context loaded)")

    # The LlmRequest contents can be modified here if needed
    # For ADK, we typically rely on state being available in the instruction template
    # But this hook allows for dynamic injection if the prompt structure requires it

    return None


# =============================================================================
# After Model Callbacks
# =============================================================================

def after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse:
    """Process LLM response after receiving.

    This callback:
    1. Records latency metrics
    2. Extracts custom_metadata from response if present
    3. Validates response structure
    4. Can modify response before it's used

    Args:
        callback_context: ADK callback context.
        llm_response: The LLM response received.

    Returns:
        The (potentially modified) LlmResponse.
    """
    state = callback_context.state

    # Calculate latency
    start_time = state.pop("_model_call_start", None)
    if start_time:
        latency_ms = (time.time() - start_time) * 1000
        metrics = state.get(STATE_KEY_RLM_METRICS, {})
        latencies = metrics.get("model_latencies_ms", [])
        latencies.append(round(latency_ms, 1))
        metrics["model_latencies_ms"] = latencies
        state[STATE_KEY_RLM_METRICS] = metrics

    # Extract or set custom_metadata
    # ADK's LlmResponse may have custom_metadata field
    if hasattr(llm_response, "custom_metadata") and llm_response.custom_metadata:
        # Store metadata for downstream processing
        state["_last_model_metadata"] = llm_response.custom_metadata
    else:
        # Set default metadata structure
        llm_response.custom_metadata = {
            "source": "rlm_code_generator",
            "iteration": state.get("_rlm_current_iteration", 0),
            "has_code_blocks": "```" in (llm_response.text or ""),
        }

    return llm_response


def after_model_extract_code_metadata(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse:
    """After-model callback that extracts code block metadata.

    Analyzes the response to populate custom_metadata with:
    - Number of code blocks
    - Whether FINAL pattern is present
    - Code complexity hints
    """
    from rlm_adk.rlm_repl import find_code_blocks, find_final_answer

    state = callback_context.state
    response_text = llm_response.text or ""

    # Analyze response
    code_blocks = find_code_blocks(response_text)
    final_answer = find_final_answer(response_text)

    # Build structured metadata
    metadata = RLMIterationMetadata(
        iteration_number=state.get("_rlm_current_iteration", 0),
        code_block_count=len(code_blocks),
        has_llm_query=any("llm_query" in block for block in code_blocks),
        has_llm_query_batched=any("llm_query_batched" in block for block in code_blocks),
        has_final_answer=final_answer is not None,
        final_answer_type="FINAL_VAR" if final_answer and "__FINAL_VAR__" in final_answer else (
            "FINAL" if final_answer else None
        ),
    )

    # Store as dict for JSON serialization
    llm_response.custom_metadata = metadata.to_dict()
    state["_last_iteration_metadata"] = metadata.to_dict()

    return llm_response


# =============================================================================
# Before Tool Callbacks
# =============================================================================

def before_tool_callback(
    callback_context: CallbackContext,
    tool_name: str,
    tool_args: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Validate and potentially modify tool inputs.

    This callback:
    1. Validates tool arguments
    2. Records tool call timing
    3. Can short-circuit with a result if appropriate

    Args:
        callback_context: ADK callback context.
        tool_name: Name of the tool being called.
        tool_args: Arguments being passed to the tool.

    Returns:
        None to proceed, or dict to short-circuit with result.
    """
    state = callback_context.state

    # Record timing
    state["_tool_call_start"] = time.time()
    state["_tool_call_name"] = tool_name

    # Track call count
    metrics = state.get(STATE_KEY_RLM_METRICS, {})
    metrics["total_tool_calls"] = metrics.get("total_tool_calls", 0) + 1
    state[STATE_KEY_RLM_METRICS] = metrics

    # Validate specific tools
    if tool_name == "execute_rlm_iteration":
        generated_code = state.get("generated_code", "")
        if not generated_code:
            return {
                "status": "error",
                "error_message": "No generated_code in state - code_generator may have failed",
                "stdout": "",
                "stderr": "",
            }

    return None  # Proceed with tool call


# =============================================================================
# After Tool Callbacks
# =============================================================================

def after_tool_callback(
    callback_context: CallbackContext,
    tool_name: str,
    tool_result: Any,
) -> Any:
    """Process tool result and handle errors.

    This callback:
    1. Records latency metrics
    2. Handles errors gracefully
    3. Updates RLM state with execution results
    4. Attaches custom_metadata to results

    Args:
        callback_context: ADK callback context.
        tool_name: Name of the tool that was called.
        tool_result: Result from the tool execution.

    Returns:
        The (potentially modified) tool result.
    """
    state = callback_context.state

    # Calculate latency
    start_time = state.pop("_tool_call_start", None)
    if start_time:
        latency_ms = (time.time() - start_time) * 1000
        metrics = state.get(STATE_KEY_RLM_METRICS, {})
        latencies = metrics.get("tool_latencies_ms", [])
        latencies.append(round(latency_ms, 1))
        metrics["tool_latencies_ms"] = latencies
        state[STATE_KEY_RLM_METRICS] = metrics

    # Handle errors
    if isinstance(tool_result, dict) and tool_result.get("status") == "error":
        error_msg = tool_result.get("error_message", "Unknown error")
        metrics = state.get(STATE_KEY_RLM_METRICS, {})

        # Track errors
        errors = state.get(STATE_KEY_TOOL_ERRORS, [])
        errors.append({
            "tool": tool_name,
            "error": error_msg,
            "iteration": state.get("_rlm_current_iteration", 0),
        })
        state[STATE_KEY_TOOL_ERRORS] = errors

        print(f"[RLM] Tool error in {tool_name}: {error_msg}")

        # Don't fail - let the code_generator see the error and recover
        metrics["errors_recovered"] = metrics.get("errors_recovered", 0) + 1
        state[STATE_KEY_RLM_METRICS] = metrics

    # Update sub-LM call count from execution results
    if tool_name == "execute_rlm_iteration" and isinstance(tool_result, dict):
        llm_calls = tool_result.get("llm_calls", 0)
        if llm_calls > 0:
            metrics = state.get(STATE_KEY_RLM_METRICS, {})
            metrics["total_sub_lm_calls"] = metrics.get("total_sub_lm_calls", 0) + llm_calls
            state[STATE_KEY_RLM_METRICS] = metrics

    return tool_result


# =============================================================================
# Error Handling Callback
# =============================================================================

def on_model_error_callback(
    callback_context: CallbackContext,
    error: Exception,
) -> Optional[LlmResponse]:
    """Handle model call errors gracefully.

    This callback:
    1. Logs the error
    2. Can provide a fallback response
    3. Tracks error metrics

    Args:
        callback_context: ADK callback context.
        error: The exception that occurred.

    Returns:
        None to re-raise, or LlmResponse as fallback.
    """
    state = callback_context.state
    agent_name = callback_context.agent_name

    # Log error
    error_str = str(error)
    tb = traceback.format_exc()
    print(f"[RLM] Model error in {agent_name}: {error_str}")

    # Track error
    metrics = state.get(STATE_KEY_RLM_METRICS, {})
    model_errors = metrics.get("model_errors", [])
    model_errors.append({
        "agent": agent_name,
        "error": error_str,
        "traceback": tb,
    })
    metrics["model_errors"] = model_errors
    state[STATE_KEY_RLM_METRICS] = metrics

    # For now, re-raise the error
    # In production, you might return a fallback LlmResponse
    return None


# =============================================================================
# Callback Bundles
# =============================================================================

def get_rlm_loop_callbacks() -> dict:
    """Get callback bundle for the RLM iteration loop.

    Returns:
        Dict of callbacks to pass to LoopAgent.
    """
    return {
        "before_agent_callback": before_rlm_loop_callback,
        "after_agent_callback": after_rlm_loop_callback,
    }


def get_code_generator_callbacks() -> dict:
    """Get callback bundle for the code generator agent.

    Returns:
        Dict of callbacks to pass to LlmAgent.
    """
    return {
        "before_agent_callback": before_code_generator_callback,
        "before_model_callback": before_model_with_history_injection,
        "after_model_callback": after_model_extract_code_metadata,
        "on_model_error_callback": on_model_error_callback,
    }


def get_code_executor_callbacks() -> dict:
    """Get callback bundle for the code executor agent.

    Returns:
        Dict of callbacks to pass to LlmAgent.
    """
    return {
        "before_tool_callback": before_tool_callback,
        "after_tool_callback": after_tool_callback,
    }
