# RLM-ADK Agent Implementation Specification

## Full RLM Integration with ADK LoopAgent

**Version:** 3.0 (Enhanced with Callbacks & System Prompt Integration)
**Status:** Approved for Implementation
**Last Updated:** 2026-01-05

---

## Executive Summary

This specification describes the implementation of a full Recursive Language Model (RLM) system using Google's Agent Development Kit (ADK). The design uses ADK's `LoopAgent` workflow to implement the iterative execution pattern that is fundamental to the RLM paradigm.

### Key Design Decisions

1. **Use `LlmAgent`** (not `Agent`) for all LLM-powered agents
2. **Implement real `llm_query()`** that calls ADK's LLM (not placeholders)
3. **Explicit iteration history** passed via state for feedback loop
4. **`BaseAgent`** for completion checker to enable escalation
5. **Nested workflow agents**: `SequentialAgent` → `LoopAgent` → sub-agents
6. **Reuse legacy RLM system prompt** from `rlm/utils/prompts.py` with domain-specific extensions
7. **ADK Callbacks** for state management, error handling, and structured metadata
8. **`custom_metadata`** for typed communication between REPL orchestration and sub-agents

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         root_agent (LlmAgent)                                │
│                         [with RLM callbacks attached]                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │              rlm_completion_workflow (SequentialAgent)                 │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │  │
│  │  │  Step 1: context_loader (LlmAgent)                                │ │  │
│  │  │  - Loads data into RLM context via tools                         │ │  │
│  │  │  - Output: context_setup_result                                  │ │  │
│  │  └──────────────────────────────────────────────────────────────────┘ │  │
│  │                                 ↓                                      │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │  │
│  │  │  Step 2: rlm_iteration_loop (LoopAgent) [max_iterations=10]      │ │  │
│  │  │  [before_agent_callback: initialize iteration state]             │ │  │
│  │  │  [after_agent_callback: finalize metrics & cleanup]              │ │  │
│  │  │                                                                   │ │  │
│  │  │    ┌────────────────────────────────────────────────────────┐    │ │  │
│  │  │    │ 2a. code_generator (LlmAgent)                          │    │ │  │
│  │  │    │     [before_model_callback: inject iteration_history]  │    │ │  │
│  │  │    │     [after_model_callback: extract custom_metadata]    │    │ │  │
│  │  │    │     - Uses: RLM_SYSTEM_PROMPT + HEALTHCARE_EXTENSION   │    │ │  │
│  │  │    │     - Sees: {iteration_history}, {context_description} │    │ │  │
│  │  │    │     - Generates Python with llm_query() calls          │    │ │  │
│  │  │    │     - Output: generated_code + custom_metadata         │    │ │  │
│  │  │    └────────────────────────────────────────────────────────┘    │ │  │
│  │  │                              ↓                                    │ │  │
│  │  │    ┌────────────────────────────────────────────────────────┐    │ │  │
│  │  │    │ 2b. code_executor (LlmAgent)                           │    │ │  │
│  │  │    │     [before_tool_callback: validate code blocks]       │    │ │  │
│  │  │    │     [after_tool_callback: capture execution metrics]   │    │ │  │
│  │  │    │     - Extracts code blocks from generated_code         │    │ │  │
│  │  │    │     - Executes in REPL with REAL llm_query bridge      │    │ │  │
│  │  │    │     - Appends to iteration_history                     │    │ │  │
│  │  │    │     - Output: execution_result + custom_metadata       │    │ │  │
│  │  │    └────────────────────────────────────────────────────────┘    │ │  │
│  │  │                              ↓                                    │ │  │
│  │  │    ┌────────────────────────────────────────────────────────┐    │ │  │
│  │  │    │ 2c. completion_checker (BaseAgent)                     │    │ │  │
│  │  │    │     - Checks for FINAL/FINAL_VAR patterns              │    │ │  │
│  │  │    │     - If found: escalate=True (exit loop)              │    │ │  │
│  │  │    │     - If not: continue to next iteration               │    │ │  │
│  │  │    └────────────────────────────────────────────────────────┘    │ │  │
│  │  │                                                                   │ │  │
│  │  │    [Loop repeats until FINAL or max_iterations]                  │ │  │
│  │  └──────────────────────────────────────────────────────────────────┘ │  │
│  │                                 ↓                                      │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐ │  │
│  │  │  Step 3: result_formatter (LlmAgent)                             │ │  │
│  │  │  - Formats rlm_final_answer for user presentation               │ │  │
│  │  │  - Output: rlm_formatted_result                                 │ │  │
│  │  └──────────────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## RLM Principles Compliance

| Principle | Implementation | Status |
|-----------|----------------|--------|
| **Context Offloading** | Data loaded into REPL `context` variable via `rlm_load_context` tool | ✅ |
| **llm_query()** | Real LLM calls via `create_llm_query_bridge()` using ADK's model | ✅ |
| **llm_query_batched()** | Concurrent calls via `asyncio.gather()` | ✅ |
| **Iterative Execution** | `LoopAgent` with explicit `iteration_history` in state | ✅ |
| **FINAL Termination** | `BaseAgent` detects patterns and escalates | ✅ |
| **Legacy System Prompt** | Import `RLM_SYSTEM_PROMPT` from `rlm/utils/prompts.py` | ✅ |
| **State Management** | ADK Callbacks manage state transitions and persistence | ✅ |
| **Error Handling** | Callbacks provide graceful error recovery | ✅ |
| **Structured Output** | `custom_metadata` enables typed REPL↔sub-agent communication | ✅ |

---

## Implementation Details

### File Structure

```
rlm_adk/
├── __init__.py
├── _compat.py                    # Existing compatibility layer
├── rlm_repl.py                   # Existing REPL environment
├── rlm_state.py                  # NEW: Iteration state management
├── llm_bridge.py                 # NEW: Real llm_query implementation
├── prompts.py                    # NEW: System prompt composition
├── callbacks.py                  # NEW: ADK callback implementations
├── metadata.py                   # NEW: custom_metadata schemas
├── agent.py                      # UPDATED: Root agent with workflows
├── agents/
│   ├── __init__.py
│   ├── code_generator.py         # NEW: LlmAgent for code generation
│   ├── code_executor.py          # NEW: LlmAgent for code execution
│   ├── completion_checker.py     # NEW: BaseAgent for FINAL detection
│   ├── context_setup.py          # NEW: LlmAgent for context loading
│   ├── result_formatter.py       # NEW: LlmAgent for result formatting
│   ├── rlm_loop.py               # NEW: LoopAgent workflow assembly
│   ├── erp_analyzer.py           # Existing
│   ├── vendor_matcher.py         # Existing
│   └── view_generator.py         # Existing
└── tools/
    ├── __init__.py
    ├── rlm_tools.py              # UPDATED: Real llm_query integration
    ├── context_loader.py         # Existing
    ├── databricks_repl.py        # Existing
    ├── unity_catalog.py          # Existing
    └── vendor_resolution.py      # Existing
```

---

## Step 1: System Prompt Composition

**File:** `rlm_adk/prompts.py`

This module imports the battle-tested `RLM_SYSTEM_PROMPT` from the legacy RLM codebase and extends it with healthcare/data-science specific instructions.

```python
"""System prompt composition for RLM-ADK integration.

This module composes the final system prompts by:
1. Importing the legacy RLM_SYSTEM_PROMPT from rlm/utils/prompts.py
2. Appending domain-specific extensions (healthcare vendor management)
3. Providing utility functions for dynamic prompt building
"""

from __future__ import annotations

from rlm.utils.prompts import RLM_SYSTEM_PROMPT, build_rlm_system_prompt, build_user_prompt

# =============================================================================
# Healthcare Data Science Extension
# =============================================================================

HEALTHCARE_VENDOR_EXTENSION = '''
## Healthcare Vendor Management Context

You are working in a healthcare vendor master data management environment with access to:

**Data Sources:**
- Multiple hospital chain ERP systems (Alpha, Beta, Gamma)
- Masterdata vendor database with golden records
- Unity Catalog volumes containing vendor data

**Domain-Specific Capabilities:**

1. **Vendor Resolution**: Match vendor instances across hospital chains to masterdata
   - Use Tax ID, DUNS number, and address for matching
   - Consider name variations and fuzzy matching
   - Assign confidence scores to matches

2. **Duplicate Detection**: Find potential duplicate vendors
   - Within a single hospital chain
   - Across multiple chains
   - Against masterdata golden records

3. **Data Quality Analysis**: Assess vendor data quality
   - Missing critical fields (Tax ID, address)
   - Inconsistent naming conventions
   - Outdated contact information

**Best Practices:**
- When analyzing vendors, prioritize Tax ID matches (most reliable)
- Use `llm_query_batched()` for parallel analysis across chains
- Chunk large vendor datasets (1000+ records) before analysis
- Aggregate findings with a final `llm_query()` summarization
'''

# =============================================================================
# Composed System Prompts
# =============================================================================

def get_rlm_system_prompt(include_healthcare_extension: bool = True) -> str:
    """Get the composed RLM system prompt.
    
    Args:
        include_healthcare_extension: Whether to append healthcare-specific context.
        
    Returns:
        Complete system prompt string.
    """
    base_prompt = RLM_SYSTEM_PROMPT
    
    if include_healthcare_extension:
        return base_prompt + "\n\n" + HEALTHCARE_VENDOR_EXTENSION
    
    return base_prompt


def get_code_generator_instruction(
    context_description: str = "",
    iteration_history: str = "(No previous iterations)",
    user_query: str = "",
) -> str:
    """Build the code generator instruction with dynamic state.
    
    This combines the RLM system prompt with current iteration state.
    
    Args:
        context_description: Description of loaded context data.
        iteration_history: Formatted history of previous iterations.
        user_query: The user's original query.
        
    Returns:
        Complete instruction for the code generator agent.
    """
    base_instruction = get_rlm_system_prompt(include_healthcare_extension=True)
    
    dynamic_section = f'''
## Current Session State

### Context Description
{context_description or "(No context loaded yet)"}

### Previous Iterations
{iteration_history}

### User Query
{user_query or "(Awaiting user query)"}

## Instructions

Based on the context and any previous execution results, write the NEXT code block.
Use the REPL environment with `llm_query()` and `llm_query_batched()` for recursive analysis.
'''
    
    return base_instruction + "\n\n" + dynamic_section


# =============================================================================
# Root Agent Instruction (Healthcare Data Scientist)
# =============================================================================

ROOT_AGENT_INSTRUCTION = '''You are an expert data scientist specializing in healthcare vendor management with RLM (Recursive Language Model) capabilities.

''' + RLM_SYSTEM_PROMPT + '''

''' + HEALTHCARE_VENDOR_EXTENSION + '''

## Available Workflows

### 1. Full RLM Workflow (RECOMMENDED for Complex Analysis)
**Delegate to:** `rlm_completion_workflow`

Use this for:
- Large-scale vendor resolution across multiple hospital chains
- Complex data analysis requiring iterative refinement
- Problems that benefit from recursive decomposition
- When the data is too large to analyze in a single pass

The workflow automatically:
1. Loads context from Unity Catalog
2. Iteratively generates and executes code with llm_query()
3. Continues until a FINAL answer is produced
4. Formats results for presentation

### 2. Direct RLM Tools (For Simple Cases)
Use these tools directly for simpler tasks:

- **rlm_load_context**: Load data into REPL context
- **rlm_execute_code**: Execute a single code block with llm_query() access
- **rlm_query_context**: Apply pre-built decomposition strategies

### 3. Pipeline Delegation
**Delegate to:** `vendor_resolution_pipeline`

Use for standard vendor resolution workflow:
1. Parallel ERP analysis across hospital chains
2. Vendor matching to masterdata
3. View generation

## When to Use RLM

Use the RLM workflow when:
- Data size exceeds what can be processed in one LLM call
- The problem requires breaking down into sub-problems
- You need to iteratively refine analysis based on intermediate results
- Concurrent analysis of independent data chunks would be beneficial

## Important Notes

- The RLM system uses REAL LLM calls for llm_query() - not simulations
- Variables persist across iterations in the REPL
- Use llm_query_batched() for concurrent processing of independent chunks
- The system will automatically terminate when FINAL() is called or max iterations reached
'''
```

---

## Step 2: ADK Callbacks for State Management & Error Handling

**File:** `rlm_adk/callbacks.py`

This module implements ADK callbacks for comprehensive control over agent execution.

```python
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

if TYPE_CHECKING:
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
    callback_context: "CallbackContext",
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
    callback_context: "CallbackContext",
) -> None:
    """Prepare state before code generation.
    
    Ensures iteration_history and context_description are available.
    """
    state = callback_context.state
    
    # Ensure iteration history is available
    if "iteration_history" not in state:
        session_id = state.get("rlm_session_id", "default")
        rlm_state = get_or_create_rlm_state(state, session_id)
        state["iteration_history"] = rlm_state.iteration_history


# =============================================================================
# After Agent Callbacks
# =============================================================================

def after_rlm_loop_callback(
    callback_context: "CallbackContext",
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
    callback_context: "CallbackContext",
    llm_request: "LlmRequest",
) -> Optional["LlmResponse"]:
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
    callback_context: "CallbackContext",
    llm_request: "LlmRequest",
) -> Optional["LlmResponse"]:
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
    callback_context: "CallbackContext",
    llm_response: "LlmResponse",
) -> "LlmResponse":
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
    callback_context: "CallbackContext",
    llm_response: "LlmResponse",
) -> "LlmResponse":
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
    callback_context: "CallbackContext",
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
    callback_context: "CallbackContext",
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
    callback_context: "CallbackContext",
    error: Exception,
) -> Optional["LlmResponse"]:
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
```

---

## Step 3: Custom Metadata Schemas

**File:** `rlm_adk/metadata.py`

Defines structured metadata types for REPL↔sub-agent communication.

```python
"""Custom metadata schemas for RLM-ADK integration.

These dataclasses define the structured metadata passed between:
- REPL orchestration layer
- Code generator agent
- Code executor agent
- Sub-LM calls

Using typed metadata enables:
1. Consistent data exchange formats
2. Easier debugging and logging
3. Metrics aggregation
4. Error tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RLMIterationMetadata:
    """Metadata for a single RLM iteration.
    
    Attached to LlmResponse.custom_metadata by after_model callbacks.
    """
    iteration_number: int
    code_block_count: int = 0
    has_llm_query: bool = False
    has_llm_query_batched: bool = False
    has_final_answer: bool = False
    final_answer_type: str | None = None  # "FINAL", "FINAL_VAR", or None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration_number": self.iteration_number,
            "code_block_count": self.code_block_count,
            "has_llm_query": self.has_llm_query,
            "has_llm_query_batched": self.has_llm_query_batched,
            "has_final_answer": self.has_final_answer,
            "final_answer_type": self.final_answer_type,
        }


@dataclass
class RLMExecutionMetadata:
    """Metadata for code execution results.
    
    Attached to tool results by after_tool callbacks.
    """
    iteration_number: int
    blocks_executed: int = 0
    llm_calls_made: int = 0
    execution_time_ms: float = 0.0
    status: str = "success"  # "success", "error", "no_code"
    error_type: str | None = None
    variables_created: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration_number": self.iteration_number,
            "blocks_executed": self.blocks_executed,
            "llm_calls_made": self.llm_calls_made,
            "execution_time_ms": self.execution_time_ms,
            "status": self.status,
            "error_type": self.error_type,
            "variables_created": self.variables_created,
        }


@dataclass
class RLMSubLMCallMetadata:
    """Metadata for individual sub-LM calls made via llm_query().
    
    Tracked by the llm_bridge to provide visibility into recursive calls.
    """
    call_index: int
    prompt_length: int
    response_length: int
    latency_ms: float
    is_batched: bool = False
    batch_size: int = 1
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "call_index": self.call_index,
            "prompt_length": self.prompt_length,
            "response_length": self.response_length,
            "latency_ms": self.latency_ms,
            "is_batched": self.is_batched,
            "batch_size": self.batch_size,
        }


@dataclass
class RLMSessionMetrics:
    """Aggregated metrics for an entire RLM session.
    
    Built from individual iteration/execution metadata.
    """
    session_id: str
    total_iterations: int = 0
    total_code_blocks: int = 0
    total_llm_query_calls: int = 0
    total_llm_query_batched_calls: int = 0
    total_sub_lm_calls: int = 0
    total_execution_time_seconds: float = 0.0
    total_model_latency_ms: float = 0.0
    errors_encountered: int = 0
    errors_recovered: int = 0
    final_answer_found: bool = False
    termination_reason: str = ""  # "FINAL", "max_iterations", "error"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_iterations": self.total_iterations,
            "total_code_blocks": self.total_code_blocks,
            "total_llm_query_calls": self.total_llm_query_calls,
            "total_llm_query_batched_calls": self.total_llm_query_batched_calls,
            "total_sub_lm_calls": self.total_sub_lm_calls,
            "total_execution_time_seconds": self.total_execution_time_seconds,
            "total_model_latency_ms": self.total_model_latency_ms,
            "errors_encountered": self.errors_encountered,
            "errors_recovered": self.errors_recovered,
            "final_answer_found": self.final_answer_found,
            "termination_reason": self.termination_reason,
        }


@dataclass  
class RLMContextMetadata:
    """Metadata describing the loaded context.
    
    Attached when context is loaded to help code generator understand the data.
    """
    context_type: str  # "dict", "list", "str", etc.
    total_size_chars: int
    chunk_count: int = 0
    chunk_sizes: list[int] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)  # e.g., ["hospital_chain_alpha", "masterdata"]
    schema_hint: dict[str, Any] | None = None  # Optional schema information
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "context_type": self.context_type,
            "total_size_chars": self.total_size_chars,
            "chunk_count": self.chunk_count,
            "chunk_sizes": self.chunk_sizes,
            "data_sources": self.data_sources,
            "schema_hint": self.schema_hint,
        }
    
    def format_for_prompt(self) -> str:
        """Format metadata as a string for inclusion in prompts."""
        parts = [
            f"Context type: {self.context_type}",
            f"Total size: {self.total_size_chars:,} characters",
        ]
        
        if self.chunk_count > 0:
            parts.append(f"Chunks: {self.chunk_count}")
            if len(self.chunk_sizes) <= 10:
                parts.append(f"Chunk sizes: {self.chunk_sizes}")
            else:
                parts.append(f"Chunk sizes: {self.chunk_sizes[:5]} ... [{len(self.chunk_sizes) - 5} more]")
        
        if self.data_sources:
            parts.append(f"Data sources: {', '.join(self.data_sources)}")
        
        return "\n".join(parts)
```

---

## Step 4: RLM State Manager

**File:** `rlm_adk/rlm_state.py`

```python
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
```

---

## Step 5: LLM Query Bridge (with Metadata Tracking)

**File:** `rlm_adk/llm_bridge.py`

```python
"""Bridge between RLM's llm_query and ADK's LLM infrastructure.

CRITICAL: This module implements REAL llm_query calls, not placeholders.
The llm_query function must return actual LLM responses for the RLM
paradigm to work correctly.

Enhanced with metadata tracking for observability.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from google.adk.agents.invocation_context import InvocationContext

from rlm_adk.metadata import RLMSubLMCallMetadata


# Global call counter for metadata tracking
_call_counter = 0
_call_metadata: list[dict[str, Any]] = []


def get_sub_lm_call_metadata() -> list[dict[str, Any]]:
    """Get metadata from all sub-LM calls in this session."""
    return _call_metadata.copy()


def reset_sub_lm_call_metadata() -> None:
    """Reset the call metadata (call at session start)."""
    global _call_counter, _call_metadata
    _call_counter = 0
    _call_metadata = []


def create_llm_query_bridge(
    invocation_context: "InvocationContext | None" = None,
    model: str = "gemini-3-pro",
    track_metadata: bool = True,
) -> Callable[[str], str]:
    """Create an llm_query function that makes real LLM calls.

    This is the CRITICAL component that enables recursive decomposition.
    The returned function MUST return actual LLM responses, not placeholders.

    Args:
        invocation_context: ADK invocation context for LLM access.
        model: Model to use for sub-LM calls.
        track_metadata: Whether to track call metadata.

    Returns:
        A synchronous llm_query(prompt) -> str function.
    """
    global _call_counter, _call_metadata

    def llm_query_with_context(prompt: str) -> str:
        """Make a sub-LM call using ADK's invocation context."""
        global _call_counter, _call_metadata
        
        start_time = time.time()
        _call_counter += 1
        call_index = _call_counter
        
        if invocation_context is None:
            response = _llm_query_fallback(prompt, model)
        else:
            try:
                # Use ADK's LLM client from the invocation context
                llm_client = invocation_context.llm

                async def _async_query():
                    response = await llm_client.generate_content_async(prompt)
                    return response.text

                # Run async call synchronously
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, _async_query())
                        response = future.result(timeout=60)
                else:
                    response = loop.run_until_complete(_async_query())

            except Exception as e:
                print(f"[llm_query] ADK call failed: {e}, trying fallback")
                response = _llm_query_fallback(prompt, model)
        
        # Track metadata
        if track_metadata:
            latency_ms = (time.time() - start_time) * 1000
            metadata = RLMSubLMCallMetadata(
                call_index=call_index,
                prompt_length=len(prompt),
                response_length=len(response),
                latency_ms=round(latency_ms, 1),
                is_batched=False,
                batch_size=1,
            )
            _call_metadata.append(metadata.to_dict())
        
        return response

    return llm_query_with_context


def create_llm_query_batched_bridge(
    invocation_context: "InvocationContext | None" = None,
    model: str = "gemini-3-pro",
    track_metadata: bool = True,
) -> Callable[[list[str]], list[str]]:
    """Create an llm_query_batched function for concurrent sub-LM calls.

    Args:
        invocation_context: ADK invocation context for LLM access.
        model: Model to use for sub-LM calls.
        track_metadata: Whether to track call metadata.

    Returns:
        A function llm_query_batched(prompts) -> list[str].
    """
    global _call_counter, _call_metadata
    single_query = create_llm_query_bridge(invocation_context, model, track_metadata=False)

    def llm_query_batched(prompts: list[str]) -> list[str]:
        """Execute multiple LLM queries concurrently."""
        global _call_counter, _call_metadata
        
        if not prompts:
            return []

        start_time = time.time()
        _call_counter += 1
        call_index = _call_counter
        batch_size = len(prompts)

        if invocation_context is None:
            results = [single_query(p) for p in prompts]
        else:
            try:
                llm_client = invocation_context.llm

                async def _async_batch():
                    tasks = [
                        llm_client.generate_content_async(prompt)
                        for prompt in prompts
                    ]
                    responses = await asyncio.gather(*tasks, return_exceptions=True)

                    results = []
                    for i, resp in enumerate(responses):
                        if isinstance(resp, Exception):
                            results.append(f"[Error in query {i}: {resp}]")
                        else:
                            results.append(resp.text)
                    return results

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, _async_batch())
                        results = future.result(timeout=120)
                else:
                    results = loop.run_until_complete(_async_batch())

            except Exception as e:
                print(f"[llm_query_batched] Batch call failed: {e}, falling back to sequential")
                results = [single_query(p) for p in prompts]

        # Track metadata
        if track_metadata:
            latency_ms = (time.time() - start_time) * 1000
            metadata = RLMSubLMCallMetadata(
                call_index=call_index,
                prompt_length=sum(len(p) for p in prompts),
                response_length=sum(len(r) for r in results),
                latency_ms=round(latency_ms, 1),
                is_batched=True,
                batch_size=batch_size,
            )
            _call_metadata.append(metadata.to_dict())

        return results

    return llm_query_batched


def _llm_query_fallback(prompt: str, model: str) -> str:
    """Fallback llm_query using direct Gemini API or simulation."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(model)
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"[llm_query_fallback] Gemini API failed: {e}")

    return _simulate_llm_response(prompt)


def _simulate_llm_response(prompt: str) -> str:
    """Simulate LLM response for development/testing.

    WARNING: This should only be used in development.
    """
    prompt_lower = prompt.lower()

    if "duplicate" in prompt_lower or "similar" in prompt_lower:
        return "Found 3 potential duplicates based on matching tax ID and similar names."

    if "summarize" in prompt_lower or "summary" in prompt_lower:
        return "Summary: The data contains vendor records from multiple hospital chains with potential duplicates."

    if "analyze" in prompt_lower:
        return "Analysis: Identified 45 confirmed matches and 23 potential matches requiring review."

    if "count" in prompt_lower:
        return "Count: 150 total records."

    return f"[Simulated response to: {prompt[:100]}...]"
```

---

## Step 6: Code Generator Agent (with Callbacks)

**File:** `rlm_adk/agents/code_generator.py`

```python
"""Code generator agent for RLM iteration loop.

This agent generates Python code that uses llm_query() for recursive
decomposition of problems. Uses the composed system prompt from
rlm/utils/prompts.py with healthcare extensions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.adk.agents import LlmAgent

from rlm_adk.callbacks import get_code_generator_callbacks
from rlm_adk.prompts import get_code_generator_instruction


def make_code_generator() -> "LlmAgent":
    """Create the code generator agent.

    Returns:
        LlmAgent that generates Python code with llm_query() calls.
    """
    from google.adk.agents import LlmAgent

    # Get the composed instruction (RLM_SYSTEM_PROMPT + healthcare extension)
    instruction = get_code_generator_instruction()
    
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
```

---

## Step 7: Code Executor Agent (with Callbacks)

**File:** `rlm_adk/agents/code_executor.py`

```python
"""Code executor agent for RLM iteration loop.

Extracts code blocks from generated_code and executes them in the
RLM REPL environment with real llm_query() access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google.adk.tools import tool

if TYPE_CHECKING:
    from google.adk.agents import LlmAgent
    from google.adk.tools.tool_context import ToolContext

from rlm_adk.callbacks import get_code_executor_callbacks
from rlm_adk.metadata import RLMExecutionMetadata


@tool
def execute_rlm_iteration(tool_context: "ToolContext") -> dict[str, Any]:
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


def make_code_executor() -> "LlmAgent":
    """Create the code executor agent.

    Returns:
        LlmAgent that executes RLM code blocks.
    """
    from google.adk.agents import LlmAgent

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
        tools=[execute_rlm_iteration],
        output_key="execution_result",
        before_tool_callback=callbacks.get("before_tool_callback"),
        after_tool_callback=callbacks.get("after_tool_callback"),
    )
```

---

## Step 8: Completion Checker (BaseAgent)

**File:** `rlm_adk/agents/completion_checker.py`

```python
"""Completion checker agent for RLM iteration loop.

Uses BaseAgent to detect FINAL/FINAL_VAR patterns and signal
loop termination via escalation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions

if TYPE_CHECKING:
    from google.adk.agents.invocation_context import InvocationContext

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
        ctx: "InvocationContext",
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
```

---

## Step 9: RLM Loop Assembly (with Callbacks)

**File:** `rlm_adk/agents/rlm_loop.py`

```python
"""RLM iteration loop using ADK LoopAgent.

Assembles the code_generator, code_executor, and completion_checker
into a LoopAgent workflow with callbacks for state management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.adk.agents import LoopAgent, SequentialAgent

from rlm_adk.agents.code_executor import make_code_executor
from rlm_adk.agents.code_generator import make_code_generator
from rlm_adk.agents.completion_checker import RLMCompletionChecker
from rlm_adk.agents.context_setup import make_context_setup_agent
from rlm_adk.agents.result_formatter import make_result_formatter
from rlm_adk.callbacks import get_rlm_loop_callbacks


def make_rlm_iteration_loop(max_iterations: int = 10) -> "LoopAgent":
    """Create the RLM iteration loop with callbacks.

    Args:
        max_iterations: Maximum iterations before forced termination.

    Returns:
        LoopAgent configured for RLM iteration.
    """
    from google.adk.agents import LoopAgent

    callbacks = get_rlm_loop_callbacks()

    return LoopAgent(
        name="rlm_iteration_loop",
        description="""Iterative RLM execution loop implementing recursive decomposition.

        Each iteration:
        1. code_generator: Generates Python code using llm_query() for sub-problems
        2. code_executor: Executes code in persistent REPL, updates iteration_history
        3. completion_checker: Detects FINAL/FINAL_VAR and escalates to exit

        Callbacks handle:
        - State initialization and cleanup
        - Metrics tracking
        - Error recovery
        """,
        max_iterations=max_iterations,
        sub_agents=[
            make_code_generator(),
            make_code_executor(),
            RLMCompletionChecker(),
        ],
        before_agent_callback=callbacks.get("before_agent_callback"),
        after_agent_callback=callbacks.get("after_agent_callback"),
    )


def make_rlm_completion_workflow(max_iterations: int = 10) -> "SequentialAgent":
    """Create the full RLM completion workflow.

    Args:
        max_iterations: Maximum iterations for the loop.

    Returns:
        SequentialAgent for complete RLM workflow.
    """
    from google.adk.agents import SequentialAgent

    return SequentialAgent(
        name="rlm_completion_workflow",
        description="""Full RLM recursive decomposition workflow.

        Complete pipeline for analyzing large datasets using the RLM paradigm:
        1. Loads context from Unity Catalog or custom sources
        2. Iteratively generates and executes code with llm_query()
        3. Continues until FINAL answer is produced
        4. Formats results for user presentation
        
        Uses composed system prompt from rlm/utils/prompts.py
        with healthcare vendor management extensions.
        """,
        sub_agents=[
            make_context_setup_agent(),
            make_rlm_iteration_loop(max_iterations),
            make_result_formatter(),
        ],
    )
```

---

## Step 10: Updated Root Agent

**File:** `rlm_adk/agent.py` (key changes)

```python
"""Root agent with RLM workflow integration."""

# Add these imports at the top
from rlm_adk.agents.rlm_loop import (
    make_rlm_completion_workflow,
    make_rlm_iteration_loop,
)
from rlm_adk.callbacks import (
    before_model_callback,
    after_model_callback,
    on_model_error_callback,
)
from rlm_adk.prompts import ROOT_AGENT_INSTRUCTION

# In the _get_agents() function:

def _get_agents():
    """Create all agents with proper ADK integration."""
    from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent

    # ... existing agent definitions ...

    # RLM completion workflow (nested LoopAgent)
    rlm_workflow = make_rlm_completion_workflow(max_iterations=10)

    # Direct access to just the iteration loop
    rlm_loop = make_rlm_iteration_loop(max_iterations=10)

    # Updated root agent with RLM workflow and callbacks
    root_agent = LlmAgent(
        name="rlm_data_scientist",
        model="gemini-3-pro",
        description="Healthcare data scientist with full RLM recursive decomposition capabilities",
        instruction=ROOT_AGENT_INSTRUCTION,  # From rlm_adk/prompts.py
        tools=[
            # Direct RLM tools for simple cases
            rlm_execute_code,
            rlm_load_context,
            rlm_query_context,
            rlm_get_session_state,
            rlm_clear_session,
            # ... other tools ...
        ],
        sub_agents=[
            rlm_workflow,                # Full RLM workflow (recommended)
            rlm_loop,                    # Just the iteration loop
            _vendor_resolution_pipeline, # Standard pipeline
            _parallel_erp_analysis,      # Parallel ERP analysis
            vendor_matcher_agent,        # Direct vendor matching
            view_generator_agent,        # Direct view generation
        ],
        # Root agent callbacks for metrics
        before_model_callback=before_model_callback,
        after_model_callback=after_model_callback,
        on_model_error_callback=on_model_error_callback,
    )

    return {
        "root": root_agent,
        "rlm_workflow": rlm_workflow,
        "rlm_loop": rlm_loop,
    }
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `rlm_adk/prompts.py` - System prompt composition
- [ ] Create `rlm_adk/callbacks.py` - ADK callback implementations
- [ ] Create `rlm_adk/metadata.py` - custom_metadata schemas
- [ ] Create `rlm_adk/rlm_state.py` - State management (with metadata support)
- [ ] Create `rlm_adk/llm_bridge.py` - Real llm_query with metadata tracking

### Phase 2: Agent Components
- [ ] Create `rlm_adk/agents/code_generator.py` (with callbacks)
- [ ] Create `rlm_adk/agents/code_executor.py` (with callbacks)
- [ ] Create `rlm_adk/agents/completion_checker.py`
- [ ] Create `rlm_adk/agents/context_setup.py`
- [ ] Create `rlm_adk/agents/result_formatter.py`

### Phase 3: Workflow Assembly
- [ ] Create `rlm_adk/agents/rlm_loop.py` - LoopAgent + SequentialAgent with callbacks
- [ ] Update `rlm_adk/agent.py` - Integrate with root agent
- [ ] Update `rlm_adk/agents/__init__.py` - Export new agents

### Phase 4: Testing
- [ ] Create `tests/rlm_adk/test_rlm_loop.py`
- [ ] Create `tests/rlm_adk/test_callbacks.py`
- [ ] Create `tests/rlm_adk/test_prompts.py`
- [ ] Create `tests/rlm_adk/test_metadata.py`
- [ ] Integration tests with mock ADK context

### Phase 5: Documentation
- [ ] Update README with RLM workflow usage
- [ ] Document callback hooks and customization
- [ ] Document custom_metadata schemas

---

## ADK Callback Reference

| Callback | When | Use Case |
|----------|------|----------|
| `before_agent_callback` | Before agent starts | Initialize state, start timers |
| `after_agent_callback` | After agent completes | Finalize metrics, cleanup |
| `before_model_callback` | Before LLM call | Inject history, validate request |
| `after_model_callback` | After LLM response | Extract metadata, track latency |
| `before_tool_callback` | Before tool execution | Validate inputs, start timing |
| `after_tool_callback` | After tool execution | Handle errors, update state |
| `on_model_error_callback` | On LLM error | Graceful recovery, logging |

---

## custom_metadata Schema Summary

| Schema | Purpose | Used By |
|--------|---------|---------|
| `RLMIterationMetadata` | Code block analysis | `after_model_callback` |
| `RLMExecutionMetadata` | Execution results | `execute_rlm_iteration` tool |
| `RLMSubLMCallMetadata` | Sub-LM call tracking | `llm_bridge` |
| `RLMSessionMetrics` | Aggregated session metrics | `after_agent_callback` |
| `RLMContextMetadata` | Context description | Context loaders |

---

## Appendix: ADK Import Reference

```python
# Agents
from google.adk.agents import LlmAgent          # LLM-powered agent
from google.adk.agents import BaseAgent         # Custom agent base class
from google.adk.agents import SequentialAgent   # Sequential workflow
from google.adk.agents import ParallelAgent     # Parallel workflow
from google.adk.agents import LoopAgent         # Iterative workflow

# Events (for BaseAgent)
from google.adk.events import Event, EventActions

# Invocation Context
from google.adk.agents.invocation_context import InvocationContext

# Callbacks
from google.adk.agents.callback_context import CallbackContext

# Models (for callbacks)
from google.adk.models import LlmRequest, LlmResponse

# Tools
from google.adk.tools import tool
from google.adk.tools.tool_context import ToolContext

# Content
from google.genai.types import Content, Part
```
