"""Tests for RLM-ADK callback implementations.

Tests all ADK callback functions for state management, metrics tracking,
and error handling. These callbacks are attached to agents at various
lifecycle points to track execution state and extract metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from rlm_adk.callbacks import (
    STATE_KEY_ITERATION_START,
    STATE_KEY_RLM_METRICS,
    STATE_KEY_TOOL_ERRORS,
    after_model_callback,
    after_model_extract_code_metadata,
    after_rlm_loop_callback,
    after_tool_callback,
    before_code_generator_callback,
    before_model_callback,
    before_model_with_history_injection,
    before_rlm_loop_callback,
    before_tool_callback,
    get_code_executor_callbacks,
    get_code_generator_callbacks,
    get_rlm_loop_callbacks,
    on_model_error_callback,
)
from rlm_adk.metadata import RLMIterationMetadata
from rlm_adk.rlm_state import RLMSessionState, get_or_create_rlm_state

# =============================================================================
# Mock Objects
# =============================================================================


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


@dataclass
class MockLlmRequest:
    """Mock ADK LlmRequest for testing."""

    prompt: str = ""
    messages: list = field(default_factory=list)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_context():
    """Create a fresh mock callback context."""
    return MockCallbackContext(state={}, agent_name="test_agent")


@pytest.fixture
def mock_context_with_session():
    """Create a mock callback context with a session ID."""
    return MockCallbackContext(
        state={"rlm_session_id": "test_session_123"}, agent_name="test_agent"
    )


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return MockLlmResponse(text="Test response", custom_metadata=None)


@pytest.fixture
def mock_llm_request():
    """Create a mock LLM request."""
    return MockLlmRequest(prompt="Test prompt")


# =============================================================================
# Test Classes
# =============================================================================


class TestBeforeAgentCallbacks:
    """Test callbacks that run before agents."""

    def test_before_rlm_loop_initializes_rlm_state(self, mock_context):
        """Creates RLMSessionState in state."""
        before_rlm_loop_callback(mock_context)

        # State key should exist
        assert "_rlm_state_default" in mock_context.state

        # Should be an RLMSessionState instance
        rlm_state = mock_context.state["_rlm_state_default"]
        assert isinstance(rlm_state, RLMSessionState)
        assert rlm_state.session_id == "default"

    def test_before_rlm_loop_initializes_metrics(self, mock_context):
        """Creates _rlm_metrics structure."""
        before_rlm_loop_callback(mock_context)

        # Metrics dict should exist
        assert STATE_KEY_RLM_METRICS in mock_context.state

        metrics = mock_context.state[STATE_KEY_RLM_METRICS]
        assert isinstance(metrics, dict)

        # Expected keys
        expected_keys = [
            "loop_start_time",
            "total_model_calls",
            "total_tool_calls",
            "total_sub_lm_calls",
            "errors_recovered",
            "model_latencies_ms",
            "tool_latencies_ms",
        ]
        for key in expected_keys:
            assert key in metrics

    def test_before_rlm_loop_records_start_time(self, mock_context):
        """Records _rlm_iteration_start_time."""
        start = time.time()
        before_rlm_loop_callback(mock_context)
        end = time.time()

        # Start time should be recorded
        assert STATE_KEY_ITERATION_START in mock_context.state

        recorded_time = mock_context.state[STATE_KEY_ITERATION_START]
        # Should be within test execution timeframe
        assert start <= recorded_time <= end

    def test_before_rlm_loop_custom_session_id(self, mock_context_with_session):
        """Uses session_id from state."""
        before_rlm_loop_callback(mock_context_with_session)

        # Should use custom session_id
        state_key = "_rlm_state_test_session_123"
        assert state_key in mock_context_with_session.state

        rlm_state = mock_context_with_session.state[state_key]
        assert rlm_state.session_id == "test_session_123"

    def test_before_code_generator_ensures_iteration_history(self, mock_context):
        """Populates iteration_history."""
        # Initialize RLM state first
        session_id = "default"
        rlm_state = get_or_create_rlm_state(mock_context.state, session_id)

        before_code_generator_callback(mock_context)

        # iteration_history should be populated
        assert "iteration_history" in mock_context.state
        assert mock_context.state["iteration_history"] == rlm_state.iteration_history

    def test_before_code_generator_ensures_context_description(self, mock_context):
        """Populates context_description."""
        before_code_generator_callback(mock_context)

        # context_description should exist
        assert "context_description" in mock_context.state
        # Should have a default value if not provided
        assert isinstance(mock_context.state["context_description"], str)

    def test_before_code_generator_sets_default_user_query(self, mock_context):
        """Sets user_query if missing."""
        before_code_generator_callback(mock_context)

        # user_query should be populated with default
        assert "user_query" in mock_context.state
        assert isinstance(mock_context.state["user_query"], str)

    def test_before_code_generator_uses_existing_values(self, mock_context):
        """Doesn't overwrite existing state."""
        # Set pre-existing values
        mock_context.state["iteration_history"] = "Custom history"
        mock_context.state["context_description"] = "Custom context"
        mock_context.state["user_query"] = "Custom query"

        before_code_generator_callback(mock_context)

        # Values should be unchanged
        assert mock_context.state["iteration_history"] == "Custom history"
        assert mock_context.state["context_description"] == "Custom context"
        assert mock_context.state["user_query"] == "Custom query"


class TestAfterAgentCallbacks:
    """Test callbacks that run after agents complete."""

    def test_after_rlm_loop_calculates_total_time(self, mock_context):
        """Adds total_execution_time_seconds."""
        # Initialize metrics with start time
        mock_context.state[STATE_KEY_RLM_METRICS] = {
            "loop_start_time": time.time() - 1.5  # 1.5 seconds ago
        }

        after_rlm_loop_callback(mock_context)

        metrics = mock_context.state[STATE_KEY_RLM_METRICS]
        assert "total_execution_time_seconds" in metrics

        # Should be a positive float around 1.5 seconds
        total_time = metrics["total_execution_time_seconds"]
        assert isinstance(total_time, (int, float))
        assert total_time > 0
        assert 1.0 <= total_time <= 2.0  # Allow some tolerance

    def test_after_rlm_loop_aggregates_iteration_count(self, mock_context):
        """Records final_iteration_count."""
        # Initialize RLM state with iterations
        session_id = "default"
        rlm_state = get_or_create_rlm_state(mock_context.state, session_id)
        rlm_state.add_iteration(
            generated_code="print('test')",
            execution_result={"stdout": "test", "stderr": "", "llm_calls": 0},
        )
        rlm_state.add_iteration(
            generated_code="print('test2')",
            execution_result={"stdout": "test2", "stderr": "", "llm_calls": 0},
        )

        mock_context.state[STATE_KEY_RLM_METRICS] = {}

        after_rlm_loop_callback(mock_context)

        metrics = mock_context.state[STATE_KEY_RLM_METRICS]
        assert metrics["final_iteration_count"] == 2

    def test_after_rlm_loop_stores_metrics_in_state(self, mock_context):
        """Sets rlm_execution_metrics."""
        mock_context.state[STATE_KEY_RLM_METRICS] = {
            "loop_start_time": time.time() - 0.5
        }

        # Initialize RLM state
        get_or_create_rlm_state(mock_context.state, "default")

        after_rlm_loop_callback(mock_context)

        # Should store metrics in state
        assert "rlm_execution_metrics" in mock_context.state
        assert isinstance(mock_context.state["rlm_execution_metrics"], dict)

    def test_after_rlm_loop_aggregates_sub_lm_calls(self, mock_context):
        """Records total_sub_lm_calls."""
        # Initialize RLM state with sub-LM calls
        session_id = "default"
        rlm_state = get_or_create_rlm_state(mock_context.state, session_id)
        rlm_state.add_iteration(
            generated_code="result = llm_query('test')",
            execution_result={"stdout": "response", "stderr": "", "llm_calls": 2},
        )
        rlm_state.add_iteration(
            generated_code="result = llm_query('test2')",
            execution_result={"stdout": "response2", "stderr": "", "llm_calls": 3},
        )

        mock_context.state[STATE_KEY_RLM_METRICS] = {}

        after_rlm_loop_callback(mock_context)

        metrics = mock_context.state[STATE_KEY_RLM_METRICS]
        # total_llm_calls should be sum of llm_calls across iterations
        assert metrics["total_sub_lm_calls"] == 5


class TestBeforeModelCallbacks:
    """Test callbacks that intercept LLM requests."""

    def test_before_model_records_timing(self, mock_context, mock_llm_request):
        """Sets _model_call_start."""
        start = time.time()
        before_model_callback(mock_context, mock_llm_request)
        end = time.time()

        # Timing should be recorded
        assert "_model_call_start" in mock_context.state

        recorded_time = mock_context.state["_model_call_start"]
        assert start <= recorded_time <= end

    def test_before_model_increments_call_count(self, mock_context, mock_llm_request):
        """Increments total_model_calls."""
        # Initialize metrics
        mock_context.state[STATE_KEY_RLM_METRICS] = {"total_model_calls": 0}

        before_model_callback(mock_context, mock_llm_request)

        metrics = mock_context.state[STATE_KEY_RLM_METRICS]
        assert metrics["total_model_calls"] == 1

        # Call again
        before_model_callback(mock_context, mock_llm_request)
        assert metrics["total_model_calls"] == 2

    def test_before_model_returns_none_to_proceed(self, mock_context, mock_llm_request):
        """Returns None (proceed with call)."""
        result = before_model_callback(mock_context, mock_llm_request)
        assert result is None

    def test_before_model_with_history_records_timing(
        self, mock_context, mock_llm_request
    ):
        """Also records timing."""
        start = time.time()
        before_model_with_history_injection(mock_context, mock_llm_request)
        end = time.time()

        # Timing should be recorded
        assert "_model_call_start" in mock_context.state
        recorded_time = mock_context.state["_model_call_start"]
        assert start <= recorded_time <= end


class TestAfterModelCallbacks:
    """Test callbacks that process LLM responses."""

    def test_after_model_calculates_latency(self, mock_context, mock_llm_response):
        """Records latency in model_latencies_ms."""
        # Set up timing
        mock_context.state["_model_call_start"] = time.time() - 0.1  # 100ms ago
        mock_context.state[STATE_KEY_RLM_METRICS] = {"model_latencies_ms": []}

        after_model_callback(mock_context, mock_llm_response)

        metrics = mock_context.state[STATE_KEY_RLM_METRICS]
        latencies = metrics["model_latencies_ms"]

        # Should have one latency entry
        assert len(latencies) == 1
        assert isinstance(latencies[0], (int, float))
        assert latencies[0] > 0

    def test_after_model_removes_start_time(self, mock_context, mock_llm_response):
        """Removes _model_call_start."""
        mock_context.state["_model_call_start"] = time.time()
        mock_context.state[STATE_KEY_RLM_METRICS] = {"model_latencies_ms": []}

        after_model_callback(mock_context, mock_llm_response)

        # Start time should be removed
        assert "_model_call_start" not in mock_context.state

    def test_after_model_returns_response(self, mock_context, mock_llm_response):
        """Returns the LlmResponse."""
        result = after_model_callback(mock_context, mock_llm_response)
        assert result is mock_llm_response

    def test_after_model_sets_default_metadata(self, mock_context):
        """Sets custom_metadata if missing."""
        response = MockLlmResponse(text="Test with ```python\ncode\n```")
        # No custom_metadata initially
        assert response.custom_metadata is None

        after_model_callback(mock_context, response)

        # Should now have metadata
        assert response.custom_metadata is not None
        assert isinstance(response.custom_metadata, dict)
        assert "source" in response.custom_metadata

    def test_after_model_preserves_existing_metadata(self, mock_context):
        """Doesn't overwrite existing metadata."""
        existing_metadata = {"custom_key": "custom_value", "iteration": 5}
        response = MockLlmResponse(text="Test", custom_metadata=existing_metadata)

        after_model_callback(mock_context, response)

        # Should preserve existing metadata
        assert response.custom_metadata == existing_metadata

    def test_after_model_extract_code_metadata_parses_blocks(self, mock_context):
        """Counts code blocks."""
        response = MockLlmResponse(
            text="""
Here's the code:

```python
x = 1
```

And another:

```python
y = 2
```
"""
        )

        after_model_extract_code_metadata(mock_context, response)

        # Should have metadata with code block count
        assert response.custom_metadata is not None
        assert response.custom_metadata["code_block_count"] == 2

    def test_after_model_extract_code_metadata_detects_llm_query(self, mock_context):
        """Sets has_llm_query flag."""
        response = MockLlmResponse(
            text="""
```python
result = llm_query("Analyze this data")
print(result)
```
"""
        )

        after_model_extract_code_metadata(mock_context, response)

        assert response.custom_metadata["has_llm_query"] is True

    def test_after_model_extract_code_metadata_detects_llm_query_batched(
        self, mock_context
    ):
        """Sets has_llm_query_batched."""
        response = MockLlmResponse(
            text="""
```python
prompts = ["Q1", "Q2", "Q3"]
results = llm_query_batched(prompts)
```
"""
        )

        after_model_extract_code_metadata(mock_context, response)

        assert response.custom_metadata["has_llm_query_batched"] is True

    def test_after_model_extract_code_metadata_detects_final(self, mock_context):
        """Sets has_final_answer."""
        response = MockLlmResponse(
            text="""
```python
FINAL("Found 42 duplicates")
```
"""
        )

        after_model_extract_code_metadata(mock_context, response)

        assert response.custom_metadata["has_final_answer"] is True
        assert response.custom_metadata["final_answer_type"] == "FINAL"

    def test_after_model_extract_code_metadata_detects_final_var(self, mock_context):
        """Detects FINAL_VAR pattern."""
        response = MockLlmResponse(
            text="""
```python
result = {"count": 42}
FINAL_VAR(result)
```
"""
        )

        after_model_extract_code_metadata(mock_context, response)

        assert response.custom_metadata["has_final_answer"] is True
        # The detection looks for __FINAL_VAR__ in the final_answer string returned by find_final_answer
        # which formats it as "__FINAL_VAR__:variable_name"
        assert response.custom_metadata["final_answer_type"] == "FINAL_VAR"


class TestBeforeToolCallbacks:
    """Test callbacks that intercept tool calls."""

    def test_before_tool_records_timing(self, mock_context):
        """Sets _tool_call_start."""
        start = time.time()
        before_tool_callback(mock_context, "test_tool", {})
        end = time.time()

        assert "_tool_call_start" in mock_context.state
        recorded_time = mock_context.state["_tool_call_start"]
        assert start <= recorded_time <= end

    def test_before_tool_records_tool_name(self, mock_context):
        """Sets _tool_call_name."""
        before_tool_callback(mock_context, "my_custom_tool", {"arg": "value"})

        assert "_tool_call_name" in mock_context.state
        assert mock_context.state["_tool_call_name"] == "my_custom_tool"

    def test_before_tool_increments_count(self, mock_context):
        """Increments total_tool_calls."""
        mock_context.state[STATE_KEY_RLM_METRICS] = {"total_tool_calls": 0}

        before_tool_callback(mock_context, "tool1", {})
        metrics = mock_context.state[STATE_KEY_RLM_METRICS]
        assert metrics["total_tool_calls"] == 1

        before_tool_callback(mock_context, "tool2", {})
        assert metrics["total_tool_calls"] == 2

    def test_before_tool_returns_none_normally(self, mock_context):
        """Returns None to proceed."""
        result = before_tool_callback(mock_context, "normal_tool", {})
        assert result is None

    def test_before_tool_validates_execute_rlm_iteration_missing_code(
        self, mock_context
    ):
        """Short-circuits if no generated_code."""
        # No generated_code in state
        result = before_tool_callback(mock_context, "execute_rlm_iteration", {})

        # Should return error dict
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "error_message" in result
        assert "generated_code" in result["error_message"]

    def test_before_tool_validates_execute_rlm_iteration_with_code(self, mock_context):
        """Proceeds if generated_code present."""
        # Set generated_code
        mock_context.state["generated_code"] = "print('hello')"

        result = before_tool_callback(mock_context, "execute_rlm_iteration", {})

        # Should proceed (return None)
        assert result is None


class TestAfterToolCallbacks:
    """Test callbacks that process tool results."""

    def test_after_tool_calculates_latency(self, mock_context):
        """Records in tool_latencies_ms."""
        mock_context.state["_tool_call_start"] = time.time() - 0.05  # 50ms ago
        mock_context.state[STATE_KEY_RLM_METRICS] = {"tool_latencies_ms": []}

        tool_result = {"status": "success", "output": "result"}
        after_tool_callback(mock_context, "test_tool", tool_result)

        metrics = mock_context.state[STATE_KEY_RLM_METRICS]
        latencies = metrics["tool_latencies_ms"]

        assert len(latencies) == 1
        assert isinstance(latencies[0], (int, float))
        assert latencies[0] > 0

    def test_after_tool_removes_start_time(self, mock_context):
        """Removes _tool_call_start."""
        mock_context.state["_tool_call_start"] = time.time()
        mock_context.state[STATE_KEY_RLM_METRICS] = {"tool_latencies_ms": []}

        after_tool_callback(mock_context, "test_tool", {"status": "success"})

        assert "_tool_call_start" not in mock_context.state

    def test_after_tool_returns_result(self, mock_context):
        """Returns the tool_result."""
        tool_result = {"status": "success", "data": [1, 2, 3]}

        result = after_tool_callback(mock_context, "test_tool", tool_result)

        assert result is tool_result

    def test_after_tool_tracks_errors(self, mock_context):
        """Adds errors to _rlm_tool_errors."""
        mock_context.state[STATE_KEY_RLM_METRICS] = {"errors_recovered": 0}

        tool_result = {
            "status": "error",
            "error_message": "NameError: variable not defined",
        }

        after_tool_callback(mock_context, "execute_rlm_iteration", tool_result)

        # Error should be tracked
        assert STATE_KEY_TOOL_ERRORS in mock_context.state
        errors = mock_context.state[STATE_KEY_TOOL_ERRORS]

        assert len(errors) == 1
        assert errors[0]["tool"] == "execute_rlm_iteration"
        assert "NameError" in errors[0]["error"]

    def test_after_tool_increments_errors_recovered(self, mock_context):
        """Increments on error."""
        mock_context.state[STATE_KEY_RLM_METRICS] = {"errors_recovered": 0}

        tool_result = {"status": "error", "error_message": "Test error"}

        after_tool_callback(mock_context, "test_tool", tool_result)

        metrics = mock_context.state[STATE_KEY_RLM_METRICS]
        assert metrics["errors_recovered"] == 1

    def test_after_tool_updates_sub_lm_count(self, mock_context):
        """Updates total_sub_lm_calls from execution."""
        mock_context.state[STATE_KEY_RLM_METRICS] = {"total_sub_lm_calls": 5}

        # Execution result with sub-LM calls
        tool_result = {
            "status": "success",
            "stdout": "output",
            "llm_calls": 3,  # Made 3 sub-LM calls
        }

        after_tool_callback(mock_context, "execute_rlm_iteration", tool_result)

        metrics = mock_context.state[STATE_KEY_RLM_METRICS]
        # Should add 3 to existing 5
        assert metrics["total_sub_lm_calls"] == 8


class TestErrorCallbacks:
    """Test error handling callbacks."""

    def test_on_model_error_tracks_error(self, mock_context):
        """Adds to model_errors list."""
        mock_context.state[STATE_KEY_RLM_METRICS] = {"model_errors": []}

        error = ValueError("API key not found")

        on_model_error_callback(mock_context, error)

        metrics = mock_context.state[STATE_KEY_RLM_METRICS]
        model_errors = metrics["model_errors"]

        assert len(model_errors) == 1
        assert model_errors[0]["agent"] == "test_agent"
        assert "API key not found" in model_errors[0]["error"]
        assert "traceback" in model_errors[0]

    def test_on_model_error_returns_none(self, mock_context):
        """Re-raises error (returns None)."""
        mock_context.state[STATE_KEY_RLM_METRICS] = {}

        error = RuntimeError("Model timeout")

        result = on_model_error_callback(mock_context, error)

        # Returns None to re-raise
        assert result is None


class TestCallbackBundles:
    """Test callback bundle factory functions."""

    def test_get_rlm_loop_callbacks_returns_dict(self):
        """Returns dict with correct keys."""
        callbacks = get_rlm_loop_callbacks()

        assert isinstance(callbacks, dict)
        assert "before_agent_callback" in callbacks
        assert "after_agent_callback" in callbacks

        # Callbacks should be callable
        assert callable(callbacks["before_agent_callback"])
        assert callable(callbacks["after_agent_callback"])

    def test_get_code_generator_callbacks_returns_dict(self):
        """Returns dict with 4 callbacks."""
        callbacks = get_code_generator_callbacks()

        assert isinstance(callbacks, dict)

        # Should have all 4 callback types
        expected_keys = [
            "before_agent_callback",
            "before_model_callback",
            "after_model_callback",
            "on_model_error_callback",
        ]
        for key in expected_keys:
            assert key in callbacks
            assert callable(callbacks[key])

    def test_get_code_executor_callbacks_returns_dict(self):
        """Returns dict with tool callbacks."""
        callbacks = get_code_executor_callbacks()

        assert isinstance(callbacks, dict)
        assert "before_tool_callback" in callbacks
        assert "after_tool_callback" in callbacks

        # Both should be callable
        assert callable(callbacks["before_tool_callback"])
        assert callable(callbacks["after_tool_callback"])
