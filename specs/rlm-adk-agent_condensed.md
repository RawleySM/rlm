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


---

## Step 2: ADK Callbacks for State Management & Error Handling

**File:** `rlm_adk/callbacks.py`

This module implements ADK callbacks for comprehensive control over agent execution.

### Callback Structure

**State Keys:**
- `STATE_KEY_RLM_METRICS`: Aggregated metrics (latencies, call counts, errors)
- `STATE_KEY_ITERATION_START`: Loop start timestamp
- `STATE_KEY_TOOL_ERRORS`: Error tracking list

**Before Agent Callbacks:**
```python
def before_rlm_loop_callback(callback_context) -> None:
    # Initialize RLM session state
    # Set up metrics tracking structure
    # Record loop start time

def before_code_generator_callback(callback_context) -> None:
    # Ensure iteration_history available in state
    # Load from RLM state if missing
```

**After Agent Callbacks:**
```python
def after_rlm_loop_callback(callback_context) -> None:
    # Calculate total execution time
    # Aggregate final metrics (iterations, sub-LM calls)
    # Store in state for result formatter
```

**Before Model Callbacks:**
```python
def before_model_callback(callback_context, llm_request) -> Optional[LlmResponse]:
    # Record timing start
    # Increment model call counter
    # Return None to proceed, or LlmResponse to short-circuit

def before_model_with_history_injection(callback_context, llm_request) -> Optional[LlmResponse]:
    # Standard timing/metrics
    # Get iteration_history and context_description from state
    # (History injection typically handled via instruction template)
```

**After Model Callbacks:**
```python
def after_model_callback(callback_context, llm_response) -> LlmResponse:
    # Calculate latency and append to metrics
    # Extract/set custom_metadata on response
    # Store metadata in state for downstream use

def after_model_extract_code_metadata(callback_context, llm_response) -> LlmResponse:
    # Parse code blocks and FINAL patterns from response
    # Build RLMIterationMetadata
    # Attach to response.custom_metadata and state
```

**Before/After Tool Callbacks:**
```python
def before_tool_callback(callback_context, tool_name, tool_args) -> Optional[dict]:
    # Record timing start
    # Validate tool inputs (e.g., check generated_code exists)
    # Return None to proceed, or dict to short-circuit

def after_tool_callback(callback_context, tool_name, tool_result) -> Any:
    # Calculate latency
    # Handle errors gracefully (track, don't fail)
    # Update sub-LM call counts from execution results
    # Attach custom_metadata to results
```

**Error Handling:**
```python
def on_model_error_callback(callback_context, error) -> Optional[LlmResponse]:
    # Log error with traceback
    # Track in metrics
    # Return None to re-raise, or LlmResponse as fallback
```

**Callback Bundles:**
```python
def get_rlm_loop_callbacks() -> dict:
    # Returns before/after_agent_callbacks for LoopAgent

def get_code_generator_callbacks() -> dict:
    # Returns before_agent, before_model, after_model, on_model_error callbacks

def get_code_executor_callbacks() -> dict:
    # Returns before_tool, after_tool callbacks
```

---

## Step 3: Custom Metadata Schemas

**File:** `rlm_adk/metadata.py`

Defines structured metadata types for REPL↔sub-agent communication. All schemas implement `to_dict()` for JSON serialization.

**Key Schemas:**

```python
@dataclass
class RLMIterationMetadata:
    """Code generation metadata attached to LlmResponse.custom_metadata."""
    iteration_number: int
    code_block_count: int = 0
    has_llm_query: bool = False
    has_llm_query_batched: bool = False
    has_final_answer: bool = False
    final_answer_type: str | None = None  # "FINAL", "FINAL_VAR", or None

@dataclass
class RLMExecutionMetadata:
    """Execution results metadata attached to tool results."""
    iteration_number: int
    blocks_executed: int = 0
    llm_calls_made: int = 0
    execution_time_ms: float = 0.0
    status: str = "success"  # "success", "error", "no_code"
    error_type: str | None = None
    variables_created: list[str] = field(default_factory=list)

@dataclass
class RLMSubLMCallMetadata:
    """Individual sub-LM call tracking from llm_bridge."""
    call_index: int
    prompt_length: int
    response_length: int
    latency_ms: float
    is_batched: bool = False
    batch_size: int = 1

@dataclass
class RLMSessionMetrics:
    """Aggregated session-level metrics."""
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

@dataclass
class RLMContextMetadata:
    """Context description metadata with format_for_prompt() helper."""
    context_type: str  # "dict", "list", "str", etc.
    total_size_chars: int
    chunk_count: int = 0
    chunk_sizes: list[int] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    schema_hint: dict[str, Any] | None = None
```

---

## Step 4: RLM State Manager

**File:** `rlm_adk/rlm_state.py`

Manages iteration history and session state across RLM loops.

**Key Classes:**

```python
@dataclass
class RLMIteration:
    """Single iteration record with format_for_prompt() method."""
    iteration_number: int
    generated_code: str
    execution_result: dict[str, Any]
    stdout: str
    stderr: str
    error: str | None = None
    llm_calls_made: int = 0
    metadata: RLMIterationMetadata | None = None
    # format_for_prompt() formats as: "=== Iteration N ===\nCode:\n...\nOutput:\n..."

@dataclass
class RLMSessionState:
    """Session-level state tracking."""
    session_id: str
    context_description: str = ""
    context_metadata: RLMContextMetadata | None = None
    iterations: list[RLMIteration] = field(default_factory=list)
    final_answer: str | None = None
    final_var_name: str | None = None
    total_llm_calls: int = 0
    
    @property
    def iteration_count(self) -> int: ...
    
    @property
    def iteration_history(self) -> str:
        # Returns formatted string of all iterations for prompt inclusion
    
    def add_iteration(...) -> RLMIteration:
        # Creates RLMIteration from code/result, appends, updates total_llm_calls
    
    def to_dict(self) -> dict: ...

def get_or_create_rlm_state(session_state: dict, session_id: str = "default") -> RLMSessionState:
    # Retrieves or creates RLM state stored at f"_rlm_state_{session_id}" in session_state
```

---

## Step 5: LLM Query Bridge (with Metadata Tracking)

**File:** `rlm_adk/llm_bridge.py`

**CRITICAL:** Implements REAL `llm_query()` calls (not placeholders) using ADK's LLM infrastructure.

**Key Functions:**

```python
def create_llm_query_bridge(
    invocation_context: InvocationContext | None = None,
    model: str = "gemini-3-pro",
    track_metadata: bool = True,
) -> Callable[[str], str]:
    """Returns synchronous llm_query(prompt) -> str function.
    
    Implementation:
    - Uses invocation_context.llm.generate_content_async() if available
    - Handles async/sync conversion (ThreadPoolExecutor if loop running)
    - Falls back to direct Gemini API or simulation if no context
    - Tracks metadata (latency, prompt/response lengths) if enabled
    """

def create_llm_query_batched_bridge(...) -> Callable[[list[str]], list[str]]:
    """Returns llm_query_batched(prompts) -> list[str] function.
    
    Implementation:
    - Uses asyncio.gather() for concurrent calls
    - Handles errors per-query (returns error strings)
    - Falls back to sequential if batch fails
    - Tracks aggregated metadata
    """

def get_sub_lm_call_metadata() -> list[dict[str, Any]]:
    """Returns metadata from all sub-LM calls in current session."""

def reset_sub_lm_call_metadata() -> None:
    """Resets call counter and metadata (call at iteration start)."""

def _llm_query_fallback(prompt: str, model: str) -> str:
    """Fallback: tries GOOGLE_API_KEY → Gemini API → simulation."""
```

**Metadata Tracking:**
- Global `_call_counter` and `_call_metadata` list
- Each call creates `RLMSubLMCallMetadata` with timing, sizes, batch info
- Metadata accessible via `get_sub_lm_call_metadata()` for metrics aggregation

---

## Step 6: Code Generator Agent (with Callbacks)

**File:** `rlm_adk/agents/code_generator.py`

Creates `LlmAgent` that generates Python code with `llm_query()` calls.

```python
def make_code_generator() -> LlmAgent:
    """Returns LlmAgent configured with:
    - Instruction: RLM_SYSTEM_PROMPT + healthcare extension (from prompts.py)
    - Model: gemini-3-pro
    - Output key: "generated_code"
    - Callbacks: before_agent, before_model (history injection), 
                 after_model (extract metadata), on_model_error
    """
```

---

## Step 7: Code Executor Agent (with Callbacks)

**File:** `rlm_adk/agents/code_executor.py`

**Tool: `execute_rlm_iteration`**
```python
@tool
def execute_rlm_iteration(tool_context: ToolContext) -> dict[str, Any]:
    """Executes generated code blocks in RLM REPL.
    
    Flow:
    1. Reset sub-LM call metadata
    2. Extract code blocks from state["generated_code"]
    3. Create llm_query bridges from invocation_context
    4. Get/create REPL session with bridges
    5. Execute each code block, collect stdout/stderr/errors
    6. Build RLMExecutionMetadata
    7. Update rlm_state.add_iteration() with results
    8. Store iteration_history in state for next iteration
    9. Return execution_result dict with custom_metadata
    """
```

**Agent Factory:**
```python
def make_code_executor() -> LlmAgent:
    """Returns LlmAgent with:
    - Instruction: Simple directive to call execute_rlm_iteration tool
    - Tools: [execute_rlm_iteration]
    - Output key: "execution_result"
    - Callbacks: before_tool, after_tool
    """
```

---

## Step 8: Completion Checker (BaseAgent)

**File:** `rlm_adk/agents/completion_checker.py`

**BaseAgent** (not LlmAgent) for deterministic FINAL pattern detection.

```python
class RLMCompletionChecker(BaseAgent):
    """Checks for FINAL/FINAL_VAR patterns and escalates to exit loop.
    
    _run_async_impl() flow:
    1. Check generated_code and execution_result.stdout for FINAL patterns
    2. If FINAL found:
       - Handle FINAL_VAR pattern (extract variable name)
       - Store final_answer in rlm_state and session_state
       - Set termination_reason = "FINAL"
       - Yield Event(actions=EventActions(escalate=True)) to exit loop
    3. If no FINAL:
       - Yield Event() to continue loop
    """
```

---

## Step 9: RLM Loop Assembly (with Callbacks)

**File:** `rlm_adk/agents/rlm_loop.py`

**LoopAgent Assembly:**
```python
def make_rlm_iteration_loop(max_iterations: int = 10) -> LoopAgent:
    """Returns LoopAgent with:
    - Sub-agents: [code_generator, code_executor, completion_checker]
    - Max iterations: 10 (default)
    - Callbacks: before_agent (init state), after_agent (finalize metrics)
    - Exits when completion_checker escalates (FINAL found) or max_iterations
    """
```

**SequentialAgent Workflow:**
```python
def make_rlm_completion_workflow(max_iterations: int = 10) -> SequentialAgent:
    """Returns SequentialAgent with 3 steps:
    1. context_setup_agent: Loads data into RLM context
    2. rlm_iteration_loop: Iterative code generation/execution
    3. result_formatter: Formats final answer for presentation
    """
```

---

## Step 10: Updated Root Agent

**File:** `rlm_adk/agent.py` (key changes)

**Integration Points:**

```python
# Imports
from rlm_adk.agents.rlm_loop import make_rlm_completion_workflow, make_rlm_iteration_loop
from rlm_adk.callbacks import before_model_callback, after_model_callback, on_model_error_callback
from rlm_adk.prompts import ROOT_AGENT_INSTRUCTION

# In _get_agents():
rlm_workflow = make_rlm_completion_workflow(max_iterations=10)
rlm_loop = make_rlm_iteration_loop(max_iterations=10)

root_agent = LlmAgent(
    name="rlm_data_scientist",
    model="gemini-3-pro",
    instruction=ROOT_AGENT_INSTRUCTION,
    tools=[rlm_execute_code, rlm_load_context, rlm_query_context, ...],
    sub_agents=[
        rlm_workflow,  # Full RLM workflow (recommended)
        rlm_loop,      # Just iteration loop
        # ... existing agents ...
    ],
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    on_model_error_callback=on_model_error_callback,
)
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
