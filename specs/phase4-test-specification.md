# Phase 4: Test Specification for RLM-ADK Agent

**Version:** 1.0
**Target Implementation:** Phase 4 of `specs/rlm-adk-agent_condensed.md`
**Status:** Ready for Implementation

---

## Overview

This document provides detailed test specifications for implementing Phase 4 (Testing) of the RLM-ADK agent integration. Tests are organized into five files as specified in the implementation checklist:

1. `tests/rlm_adk/test_rlm_loop.py` - LoopAgent and SequentialAgent workflow tests
2. `tests/rlm_adk/test_callbacks.py` - ADK callback implementations
3. `tests/rlm_adk/test_prompts.py` - System prompt composition
4. `tests/rlm_adk/test_metadata.py` - custom_metadata schemas
5. `tests/rlm_adk/test_rlm_loop_integration.py` - Integration tests with mock ADK context

---

## General Testing Guidelines

### Import Pattern

All tests requiring ADK should use the conditional skip pattern:

```python
import pytest
from rlm_adk._compat import ADK_AVAILABLE

requires_adk = pytest.mark.skipif(
    not ADK_AVAILABLE,
    reason="google-adk not installed"
)
```

### Fixture Conventions

- Clear all environment variables that could affect tests (`GOOGLE_API_KEY`, etc.)
- Use unique session IDs for each test to avoid state pollution
- Clean up REPL sessions after tests using `clear_repl_session()`

### Mock Patterns

Tests should use mock LLM functions rather than real API calls:

```python
@pytest.fixture
def mock_llm_query():
    call_count = [0]
    def llm_query(prompt: str) -> str:
        call_count[0] += 1
        return f"Mock response {call_count[0]}: {prompt[:30]}..."
    llm_query.call_count = call_count
    return llm_query
```

---

## File 1: `tests/rlm_adk/test_rlm_loop.py`

### Purpose
Test the RLM iteration loop (`LoopAgent`) and completion workflow (`SequentialAgent`) assembly.

### Test Classes

#### `class TestRLMIterationLoopStructure`
Tests that do NOT require ADK - verify factory functions and configuration.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_make_rlm_iteration_loop_returns_loop_agent` | Factory returns LoopAgent | `isinstance(result, LoopAgent)` |
| `test_loop_agent_has_correct_name` | Name is `rlm_iteration_loop` | `loop.name == "rlm_iteration_loop"` |
| `test_loop_agent_has_three_sub_agents` | Has code_generator, code_executor, completion_checker | `len(loop.sub_agents) == 3` |
| `test_loop_agent_max_iterations_default` | Default max_iterations is 10 | `loop.max_iterations == 10` |
| `test_loop_agent_max_iterations_custom` | Custom max_iterations respected | `make_rlm_iteration_loop(5).max_iterations == 5` |
| `test_loop_agent_has_before_callback` | before_agent_callback attached | `loop.before_agent_callback is not None` |
| `test_loop_agent_has_after_callback` | after_agent_callback attached | `loop.after_agent_callback is not None` |

#### `class TestRLMCompletionWorkflowStructure` (requires_adk)
Tests for the full SequentialAgent workflow.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_make_rlm_completion_workflow_returns_sequential_agent` | Factory returns SequentialAgent | `isinstance(result, SequentialAgent)` |
| `test_workflow_has_correct_name` | Name is `rlm_completion_workflow` | `workflow.name == "rlm_completion_workflow"` |
| `test_workflow_has_three_stages` | Has context_loader, loop, formatter | `len(workflow.sub_agents) == 3` |
| `test_workflow_first_stage_is_context_loader` | First stage loads context | `workflow.sub_agents[0].name == "rlm_context_loader"` |
| `test_workflow_second_stage_is_iteration_loop` | Second stage is LoopAgent | `workflow.sub_agents[1].name == "rlm_iteration_loop"` |
| `test_workflow_third_stage_is_result_formatter` | Third stage formats results | `workflow.sub_agents[2].name == "rlm_result_formatter"` |

#### `class TestSubAgentFactories` (requires_adk)
Test individual agent factory functions.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_make_code_generator_returns_llm_agent` | Factory returns LlmAgent | `isinstance(result, LlmAgent)` |
| `test_code_generator_has_output_key` | Output key is `generated_code` | `agent.output_key == "generated_code"` |
| `test_code_generator_instruction_contains_rlm_prompt` | Uses RLM system prompt | `"llm_query" in agent.instruction` |
| `test_code_generator_has_callbacks` | All 4 callbacks attached | Check before_agent, before_model, after_model, on_model_error |
| `test_make_code_executor_returns_llm_agent` | Factory returns LlmAgent | `isinstance(result, LlmAgent)` |
| `test_code_executor_has_execute_tool` | Has execute_rlm_iteration tool | Tool in `agent.tools` |
| `test_code_executor_has_output_key` | Output key is `execution_result` | `agent.output_key == "execution_result"` |
| `test_code_executor_has_tool_callbacks` | Has before_tool and after_tool callbacks | Callbacks not None |
| `test_completion_checker_is_base_agent` | RLMCompletionChecker extends BaseAgent | `isinstance(result, BaseAgent)` |
| `test_completion_checker_has_correct_name` | Default name is `rlm_completion_checker` | `checker.name == "rlm_completion_checker"` |
| `test_make_context_setup_agent_returns_llm_agent` | Factory returns LlmAgent | `isinstance(result, LlmAgent)` |
| `test_context_setup_has_load_tool` | Has rlm_load_context tool | Tool in `agent.tools` |
| `test_make_result_formatter_returns_llm_agent` | Factory returns LlmAgent | `isinstance(result, LlmAgent)` |
| `test_result_formatter_has_output_key` | Output key is `rlm_formatted_result` | `agent.output_key == "rlm_formatted_result"` |

---

## File 2: `tests/rlm_adk/test_callbacks.py`

### Purpose
Test all ADK callback implementations for state management, metrics tracking, and error handling.

### Test Classes

#### `class TestBeforeAgentCallbacks`
Test callbacks that run before agents.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_before_rlm_loop_initializes_rlm_state` | Creates RLMSessionState in state | State key `_rlm_state_default` exists |
| `test_before_rlm_loop_initializes_metrics` | Creates `_rlm_metrics` structure | Metrics dict has expected keys |
| `test_before_rlm_loop_records_start_time` | Records `_rlm_iteration_start_time` | Timestamp is recent |
| `test_before_rlm_loop_custom_session_id` | Uses session_id from state | Correct state key used |
| `test_before_code_generator_ensures_iteration_history` | Populates `iteration_history` | Key exists in state |
| `test_before_code_generator_ensures_context_description` | Populates `context_description` | Key exists in state |
| `test_before_code_generator_sets_default_user_query` | Sets `user_query` if missing | Default value present |
| `test_before_code_generator_uses_existing_values` | Doesn't overwrite existing state | Values unchanged |

#### `class TestAfterAgentCallbacks`
Test callbacks that run after agents complete.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_after_rlm_loop_calculates_total_time` | Adds `total_execution_time_seconds` | Value is positive float |
| `test_after_rlm_loop_aggregates_iteration_count` | Records `final_iteration_count` | Matches rlm_state.iteration_count |
| `test_after_rlm_loop_stores_metrics_in_state` | Sets `rlm_execution_metrics` | Dict present in state |
| `test_after_rlm_loop_aggregates_sub_lm_calls` | Records `total_sub_lm_calls` | Matches rlm_state.total_llm_calls |

#### `class TestBeforeModelCallbacks`
Test callbacks that intercept LLM requests.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_before_model_records_timing` | Sets `_model_call_start` | Timestamp in state |
| `test_before_model_increments_call_count` | Increments `total_model_calls` | Count increases |
| `test_before_model_returns_none_to_proceed` | Returns None (proceed with call) | `result is None` |
| `test_before_model_with_history_records_timing` | Also records timing | Timestamp in state |

#### `class TestAfterModelCallbacks`
Test callbacks that process LLM responses.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_after_model_calculates_latency` | Records latency in `model_latencies_ms` | List has new entry |
| `test_after_model_removes_start_time` | Removes `_model_call_start` | Key not in state |
| `test_after_model_returns_response` | Returns the LlmResponse | `result is llm_response` |
| `test_after_model_sets_default_metadata` | Sets `custom_metadata` if missing | Dict present on response |
| `test_after_model_preserves_existing_metadata` | Doesn't overwrite existing metadata | Original metadata preserved |
| `test_after_model_extract_code_metadata_parses_blocks` | Counts code blocks | `code_block_count` matches |
| `test_after_model_extract_code_metadata_detects_llm_query` | Sets `has_llm_query` flag | Flag is True when present |
| `test_after_model_extract_code_metadata_detects_llm_query_batched` | Sets `has_llm_query_batched` | Flag is True when present |
| `test_after_model_extract_code_metadata_detects_final` | Sets `has_final_answer` | Flag is True for FINAL() |
| `test_after_model_extract_code_metadata_detects_final_var` | Detects FINAL_VAR pattern | `final_answer_type == "FINAL_VAR"` |

#### `class TestBeforeToolCallbacks`
Test callbacks that intercept tool calls.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_before_tool_records_timing` | Sets `_tool_call_start` | Timestamp in state |
| `test_before_tool_records_tool_name` | Sets `_tool_call_name` | Name in state |
| `test_before_tool_increments_count` | Increments `total_tool_calls` | Count increases |
| `test_before_tool_returns_none_normally` | Returns None to proceed | `result is None` |
| `test_before_tool_validates_execute_rlm_iteration_missing_code` | Short-circuits if no generated_code | Returns error dict |
| `test_before_tool_validates_execute_rlm_iteration_with_code` | Proceeds if generated_code present | Returns None |

#### `class TestAfterToolCallbacks`
Test callbacks that process tool results.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_after_tool_calculates_latency` | Records in `tool_latencies_ms` | List has new entry |
| `test_after_tool_removes_start_time` | Removes `_tool_call_start` | Key not in state |
| `test_after_tool_returns_result` | Returns the tool_result | `result is tool_result` |
| `test_after_tool_tracks_errors` | Adds errors to `_rlm_tool_errors` | Error recorded |
| `test_after_tool_increments_errors_recovered` | Increments on error | Count increases |
| `test_after_tool_updates_sub_lm_count` | Updates `total_sub_lm_calls` from execution | Count matches |

#### `class TestErrorCallbacks`
Test error handling callbacks.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_on_model_error_tracks_error` | Adds to `model_errors` list | Error recorded with traceback |
| `test_on_model_error_returns_none` | Re-raises error (returns None) | `result is None` |

#### `class TestCallbackBundles`
Test callback bundle factory functions.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_get_rlm_loop_callbacks_returns_dict` | Returns dict with correct keys | Has `before_agent_callback`, `after_agent_callback` |
| `test_get_code_generator_callbacks_returns_dict` | Returns dict with 4 callbacks | Has all 4 callback keys |
| `test_get_code_executor_callbacks_returns_dict` | Returns dict with tool callbacks | Has `before_tool_callback`, `after_tool_callback` |

### Mock Objects Required

```python
@dataclass
class MockCallbackContext:
    """Mock ADK CallbackContext for testing."""
    state: dict = field(default_factory=dict)
    agent_name: str = "test_agent"

@dataclass  
class MockLlmResponse:
    """Mock ADK LlmResponse for testing."""
    text: str = ""
    custom_metadata: dict | None = None
```

---

## File 3: `tests/rlm_adk/test_prompts.py`

### Purpose
Test system prompt composition and dynamic prompt building.

### Test Classes

#### `class TestSystemPromptComposition`
Test the base system prompt composition.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_get_rlm_system_prompt_includes_base` | Contains RLM_SYSTEM_PROMPT content | `"llm_query" in prompt` and `"context" in prompt` |
| `test_get_rlm_system_prompt_with_healthcare_extension` | Contains healthcare content when enabled | `"Healthcare" in prompt` and `"vendor" in prompt.lower()` |
| `test_get_rlm_system_prompt_without_healthcare_extension` | Excludes healthcare when disabled | `"Healthcare" not in prompt` |
| `test_rlm_system_prompt_mentions_llm_query` | Documents llm_query function | `"llm_query(" in prompt` |
| `test_rlm_system_prompt_mentions_llm_query_batched` | Documents llm_query_batched | `"llm_query_batched(" in prompt` |
| `test_rlm_system_prompt_mentions_final` | Documents FINAL pattern | `"FINAL(" in prompt` |
| `test_rlm_system_prompt_mentions_final_var` | Documents FINAL_VAR pattern | `"FINAL_VAR(" in prompt` |
| `test_rlm_system_prompt_mentions_context` | Documents context variable | `"context" in prompt` |

#### `class TestCodeGeneratorInstruction`
Test dynamic code generator instruction building.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_get_code_generator_instruction_returns_string` | Returns non-empty string | `isinstance(result, str)` and `len(result) > 0` |
| `test_code_generator_instruction_includes_base_prompt` | Contains base RLM prompt | `"llm_query" in instruction` |
| `test_code_generator_instruction_includes_healthcare` | Contains healthcare extension | `"Healthcare" in instruction` |
| `test_code_generator_instruction_includes_context_placeholder` | Has context_description placeholder | Either placeholder or actual description present |
| `test_code_generator_instruction_includes_history_placeholder` | Has iteration_history placeholder | Either placeholder or actual history present |
| `test_code_generator_instruction_includes_query_placeholder` | Has user_query placeholder | Either placeholder or actual query present |
| `test_code_generator_instruction_with_actual_context` | Includes actual context when provided | Actual description in output |
| `test_code_generator_instruction_with_actual_history` | Includes actual history when provided | Actual history in output |
| `test_code_generator_instruction_empty_values_use_defaults` | Empty strings use default placeholders | Default text present |

#### `class TestRootAgentInstruction`
Test the root agent instruction constant.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_root_agent_instruction_exists` | ROOT_AGENT_INSTRUCTION is defined | Variable is not None |
| `test_root_agent_instruction_includes_rlm_prompt` | Contains RLM system prompt | `"llm_query" in ROOT_AGENT_INSTRUCTION` |
| `test_root_agent_instruction_includes_healthcare` | Contains healthcare extension | `"Healthcare" in ROOT_AGENT_INSTRUCTION` |
| `test_root_agent_instruction_documents_workflows` | Documents available workflows | `"rlm_completion_workflow" in ROOT_AGENT_INSTRUCTION` |
| `test_root_agent_instruction_documents_tools` | Documents direct RLM tools | `"rlm_load_context" in ROOT_AGENT_INSTRUCTION` |

#### `class TestHealthcareExtension`
Test the healthcare vendor extension content.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_healthcare_extension_mentions_hospital_chains` | Mentions Alpha, Beta, Gamma | All three chains mentioned |
| `test_healthcare_extension_mentions_masterdata` | Mentions masterdata | `"masterdata" in extension.lower()` |
| `test_healthcare_extension_mentions_tax_id` | Mentions Tax ID for matching | `"tax id" in extension.lower()` |
| `test_healthcare_extension_mentions_duns` | Mentions DUNS number | `"duns" in extension.lower()` |
| `test_healthcare_extension_mentions_chunking` | Recommends chunking large datasets | `"chunk" in extension.lower()` |

---

## File 4: `tests/rlm_adk/test_metadata.py`

### Purpose
Test custom_metadata dataclass schemas and serialization.

### Test Classes

#### `class TestRLMIterationMetadata`
Test iteration metadata schema.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_rlm_iteration_metadata_defaults` | Default values are correct | All defaults as specified |
| `test_rlm_iteration_metadata_all_fields` | All fields assignable | No exceptions |
| `test_rlm_iteration_metadata_to_dict` | Serializes to dict correctly | All keys present |
| `test_rlm_iteration_metadata_to_dict_includes_iteration_number` | Dict has iteration_number | Key present with correct value |
| `test_rlm_iteration_metadata_final_answer_types` | final_answer_type accepts valid values | "FINAL", "FINAL_VAR", None |

#### `class TestRLMExecutionMetadata`
Test execution metadata schema.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_rlm_execution_metadata_defaults` | Default values are correct | `status == "success"`, lists empty |
| `test_rlm_execution_metadata_all_fields` | All fields assignable | No exceptions |
| `test_rlm_execution_metadata_to_dict` | Serializes to dict correctly | All keys present |
| `test_rlm_execution_metadata_variables_created_mutable` | variables_created list is mutable | Can append items |
| `test_rlm_execution_metadata_status_values` | status accepts valid values | "success", "error", "no_code" |

#### `class TestRLMSubLMCallMetadata`
Test sub-LM call metadata schema.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_rlm_sub_lm_call_metadata_defaults` | Default values for batching | `is_batched == False`, `batch_size == 1` |
| `test_rlm_sub_lm_call_metadata_all_fields` | All fields assignable | No exceptions |
| `test_rlm_sub_lm_call_metadata_to_dict` | Serializes correctly | All keys present |
| `test_rlm_sub_lm_call_metadata_batch_tracking` | Tracks batch information | `is_batched` and `batch_size` reflect actual values |

#### `class TestRLMSessionMetrics`
Test session-level metrics schema.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_rlm_session_metrics_requires_session_id` | session_id is required | TypeError without it |
| `test_rlm_session_metrics_defaults` | All counters default to 0 | All numeric fields are 0 |
| `test_rlm_session_metrics_to_dict` | Serializes correctly | All keys present |
| `test_rlm_session_metrics_termination_reasons` | termination_reason accepts values | "FINAL", "max_iterations", "error" |
| `test_rlm_session_metrics_aggregation` | Fields can be incremented | Values update correctly |

#### `class TestRLMContextMetadata`
Test context description metadata schema.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_rlm_context_metadata_required_fields` | context_type and total_size_chars required | TypeError without them |
| `test_rlm_context_metadata_defaults` | List fields default to empty | `chunk_sizes == []`, `data_sources == []` |
| `test_rlm_context_metadata_to_dict` | Serializes correctly | All keys present |
| `test_rlm_context_metadata_format_for_prompt_basic` | Basic formatting works | Contains type and size |
| `test_rlm_context_metadata_format_for_prompt_with_chunks` | Includes chunk info | Contains chunk count |
| `test_rlm_context_metadata_format_for_prompt_with_sources` | Includes data sources | Sources listed |
| `test_rlm_context_metadata_format_for_prompt_truncates_many_chunks` | Truncates long chunk lists | Shows "more" indicator |
| `test_rlm_context_metadata_schema_hint_optional` | schema_hint can be None or dict | Both work |

---

## File 5: `tests/rlm_adk/test_rlm_loop_integration.py`

### Purpose
Integration tests for the full RLM loop with mock ADK context.

### Mock ADK Context

```python
@dataclass
class MockInvocationContext:
    """Mock ADK InvocationContext for testing."""
    session: "MockSession"
    llm: "MockLLMClient"

@dataclass
class MockSession:
    """Mock ADK Session."""
    state: dict = field(default_factory=dict)
    id: str = "test_session"

class MockLLMClient:
    """Mock LLM client that returns predictable responses."""
    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []
        self.call_index = 0
        
    async def generate_content_async(self, prompt: str):
        if self.call_index < len(self.responses):
            response = self.responses[self.call_index]
            self.call_index += 1
        else:
            response = f"Mock response to: {prompt[:50]}..."
        return MockGenerateResponse(text=response)

@dataclass
class MockGenerateResponse:
    text: str
```

### Test Classes

#### `class TestLlmBridgeIntegration`
Test the llm_query bridge with mock context.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_create_llm_query_bridge_with_context` | Creates bridge with invocation context | Returns callable |
| `test_llm_query_bridge_calls_llm` | Bridge calls mock LLM | Response matches mock |
| `test_llm_query_bridge_tracks_metadata` | Metadata tracked when enabled | `get_sub_lm_call_metadata()` has entries |
| `test_llm_query_bridge_metadata_has_latency` | Metadata includes latency | `latency_ms` key present |
| `test_llm_query_bridge_without_context_uses_fallback` | Falls back when no context | Returns simulated response |
| `test_create_llm_query_batched_bridge` | Creates batched bridge | Returns callable |
| `test_llm_query_batched_concurrent_calls` | Batched calls made concurrently | All responses returned |
| `test_llm_query_batched_handles_errors` | Individual errors don't fail batch | Error messages in results |
| `test_reset_sub_lm_call_metadata` | Resets call counter and metadata | Both cleared |

#### `class TestRLMStateIntegration`
Test RLM state management across iterations.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_get_or_create_rlm_state_creates_new` | Creates new state when missing | State created |
| `test_get_or_create_rlm_state_returns_existing` | Returns existing state | Same object returned |
| `test_rlm_session_state_add_iteration` | add_iteration creates RLMIteration | Iteration appended |
| `test_rlm_session_state_iteration_count_increments` | iteration_count increments | Count matches iterations |
| `test_rlm_session_state_iteration_history_formats` | iteration_history formats correctly | Contains code and output |
| `test_rlm_session_state_total_llm_calls_aggregates` | total_llm_calls sums across iterations | Total is sum |
| `test_rlm_iteration_format_for_prompt` | RLMIteration.format_for_prompt() works | Contains all sections |

#### `class TestCompletionCheckerIntegration`
Test completion checker with state.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_completion_checker_detects_final_in_code` | Finds FINAL() in generated_code | final_answer set, escalate event |
| `test_completion_checker_detects_final_in_stdout` | Finds FINAL() in execution stdout | final_answer set |
| `test_completion_checker_detects_final_var` | Handles FINAL_VAR pattern | final_var_name set |
| `test_completion_checker_continues_without_final` | No escalation without FINAL | No escalate action |
| `test_completion_checker_continues_on_error` | Continues loop on execution error | No escalate action |
| `test_completion_checker_stores_termination_reason` | Sets `rlm_termination_reason` | Value is "FINAL" |

#### `class TestExecuteRlmIterationIntegration`
Test the execute_rlm_iteration tool integration.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_execute_rlm_iteration_extracts_code_blocks` | Extracts code from generated_code | Code executed |
| `test_execute_rlm_iteration_runs_in_repl` | Code runs in REPL environment | Variables accessible |
| `test_execute_rlm_iteration_llm_query_available` | llm_query callable in code | Sub-LM call counted |
| `test_execute_rlm_iteration_llm_query_batched_available` | llm_query_batched callable | Batch call counted |
| `test_execute_rlm_iteration_context_available` | context variable accessible | Value matches loaded |
| `test_execute_rlm_iteration_updates_iteration_history` | Updates state[iteration_history] | New iteration in history |
| `test_execute_rlm_iteration_returns_metadata` | Returns custom_metadata | RLMExecutionMetadata present |
| `test_execute_rlm_iteration_handles_multiple_blocks` | Executes all code blocks | All blocks run |
| `test_execute_rlm_iteration_handles_error` | Error in code returns error status | `status == "error"` |
| `test_execute_rlm_iteration_no_code_blocks` | No code blocks returns no_code | `status == "no_code"` |

#### `class TestFullWorkflowIntegration` (requires_adk)
End-to-end workflow tests.

| Test Name | Description | Assertions |
|-----------|-------------|------------|
| `test_workflow_context_flows_through_agents` | Context loaded by first agent accessible by second | Context preserved |
| `test_workflow_iteration_history_accumulates` | History grows with each iteration | Multiple iterations recorded |
| `test_workflow_terminates_on_final` | Loop exits when FINAL detected | Escalation stops loop |
| `test_workflow_terminates_on_max_iterations` | Loop exits at max_iterations | Iteration count at max |
| `test_workflow_metrics_aggregated` | Final metrics include all iterations | Totals match sum |
| `test_workflow_final_answer_passed_to_formatter` | rlm_final_answer available to formatter | Value in state |

---

## Implementation Notes

### Creating Mock CallbackContext

The `CallbackContext` is provided by ADK and contains:
- `state`: The session state dict
- `agent_name`: Name of the current agent

For testing without ADK:

```python
@dataclass
class MockCallbackContext:
    state: dict = field(default_factory=dict)
    agent_name: str = "test_agent"
```

### Creating Mock ToolContext

Use the existing `SimpleToolContext` from `rlm_adk._compat`:

```python
from rlm_adk._compat import SimpleToolContext

ctx = SimpleToolContext({"rlm_session_id": "test_session"})
```

### Testing Async Code

The completion checker uses async generators. Use `pytest-asyncio`:

```python
import pytest

@pytest.mark.asyncio
async def test_completion_checker():
    checker = RLMCompletionChecker()
    events = []
    async for event in checker._run_async_impl(mock_ctx):
        events.append(event)
    assert len(events) == 1
```

### Environment Variable Cleanup

Use `monkeypatch` fixture to clear environment:

```python
@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
```

### Session Cleanup

Clear REPL sessions to prevent test pollution:

```python
@pytest.fixture(autouse=True)
def clean_sessions():
    yield
    from rlm_adk.rlm_repl import _ACTIVE_REPL_SESSIONS
    _ACTIVE_REPL_SESSIONS.clear()
```

---

## Checklist

### File Creation
- [ ] Create `tests/rlm_adk/test_rlm_loop.py`
- [ ] Create `tests/rlm_adk/test_callbacks.py`
- [ ] Create `tests/rlm_adk/test_prompts.py`
- [ ] Create `tests/rlm_adk/test_metadata.py`
- [ ] Create `tests/rlm_adk/test_rlm_loop_integration.py`

### Test Categories Covered
- [ ] Unit tests for all metadata dataclasses
- [ ] Unit tests for all callback functions
- [ ] Unit tests for prompt composition
- [ ] Unit tests for factory functions
- [ ] Integration tests for llm_bridge
- [ ] Integration tests for state management
- [ ] Integration tests for completion checker
- [ ] Integration tests for execute_rlm_iteration
- [ ] End-to-end workflow tests (with ADK)

### Quality Checks
- [ ] All tests pass with `uv run pytest tests/rlm_adk/`
- [ ] Tests work both with and without ADK installed
- [ ] No environment variable leakage between tests
- [ ] No session state leakage between tests
- [ ] Clear test naming following existing patterns

---

## Appendix: Test Data Examples

### Mock Generated Code

```python
MOCK_GENERATED_CODE_WITH_LLM_QUERY = '''
```python
# Analyze the data
result = llm_query("Summarize the vendors in context")
print(f"Analysis: {result}")
```
'''

MOCK_GENERATED_CODE_WITH_FINAL = '''
```python
FINAL("Found 42 duplicate vendors")
```
'''

MOCK_GENERATED_CODE_WITH_FINAL_VAR = '''
```python
result_data = {"duplicates": 42, "matches": 150}
FINAL_VAR(result_data)
```
'''

MOCK_GENERATED_CODE_MULTIPLE_BLOCKS = '''
First, load the data:

```python
vendors = context["vendors"]
print(f"Loaded {len(vendors)} vendors")
```

Then analyze:

```python
analysis = llm_query(f"Find duplicates in: {vendors[:5]}")
print(analysis)
```
'''
```

### Mock Execution Results

```python
MOCK_EXECUTION_SUCCESS = {
    "status": "success",
    "stdout": "Analysis: Found 3 potential duplicates",
    "stderr": "",
    "llm_calls": 1,
    "blocks_executed": 1,
}

MOCK_EXECUTION_ERROR = {
    "status": "error",
    "error_message": "NameError: name 'undefined_var' is not defined",
    "error_type": "NameError",
    "stdout": "",
    "stderr": "",
    "llm_calls": 0,
}
```

