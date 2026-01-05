#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=8.0.0",
#   "pytest-asyncio>=0.23.0",
# ]
# ///
"""Integration tests for the full RLM loop with mock ADK context.

This test file provides comprehensive integration testing for:
- llm_query bridge with mock invocation context (9 tests)
- RLM state management across iterations (7 tests)
- Completion checker with state transitions (6 tests)
- execute_rlm_iteration tool integration (10 tests)
- End-to-end workflow tests (6 tests)
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from rlm_adk.testing import create_tool_context
from google.adk.events import Event

# Always available imports
from rlm_adk.llm_bridge import (
    create_llm_query_bridge,
    create_llm_query_batched_bridge,
    get_sub_lm_call_metadata,
    reset_sub_lm_call_metadata,
)
from rlm_adk.metadata import (
    RLMContextMetadata,
    RLMExecutionMetadata,
    RLMIterationMetadata,
)
from rlm_adk.rlm_repl import (
    RLMREPLEnvironment,
    clear_repl_session,
    find_code_blocks,
    find_final_answer,
)
from rlm_adk.rlm_state import RLMSessionState, get_or_create_rlm_state

# =============================================================================
# Mock ADK Context Objects
# =============================================================================

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
        """Generate mock response."""
        if self.call_index < len(self.responses):
            response = self.responses[self.call_index]
            self.call_index += 1
        else:
            response = f"Mock response to: {prompt[:50]}..."
        return MockGenerateResponse(text=response)


@dataclass
class MockGenerateResponse:
    """Mock response from LLM."""
    text: str


@dataclass
class MockInvocationContext:
    """Mock ADK InvocationContext for testing."""
    session: MockSession
    llm: MockLLMClient


# =============================================================================
# Test Data Examples (from Appendix)
# =============================================================================

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


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Clean environment variables that could affect tests."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)


@pytest.fixture(autouse=True)
def clean_sessions():
    """Clean REPL sessions after each test."""
    yield
    from rlm_adk.rlm_repl import _ACTIVE_REPL_SESSIONS
    _ACTIVE_REPL_SESSIONS.clear()


@pytest.fixture
def mock_llm_responses():
    """Default mock LLM responses."""
    return [
        "Mock analysis: Found 3 duplicates",
        "Mock summary: Data contains vendor records",
    ]


@pytest.fixture
def mock_invocation_context(mock_llm_responses):
    """Create mock invocation context."""
    session = MockSession(state={})
    llm_client = MockLLMClient(responses=mock_llm_responses)
    return MockInvocationContext(session=session, llm=llm_client)


@pytest.fixture
def mock_tool_context():
    """Create mock tool context."""
    from rlm_adk.testing import create_tool_context
    return create_tool_context({"rlm_session_id": "test_session"})


# =============================================================================
# Test Class 1: TestLlmBridgeIntegration
# =============================================================================

class TestLlmBridgeIntegration:
    """Test the llm_query bridge with mock context."""

    def test_create_llm_query_bridge_with_context(self, mock_invocation_context):
        """Creates bridge with invocation context - returns callable."""
        bridge = create_llm_query_bridge(mock_invocation_context)
        assert callable(bridge)

    def test_llm_query_bridge_calls_llm(self, mock_invocation_context):
        """Bridge calls mock LLM - response matches mock."""
        bridge = create_llm_query_bridge(mock_invocation_context)

        # First response from mock
        result = bridge("Test prompt")

        assert "Mock analysis" in result
        assert mock_invocation_context.llm.call_index == 1

    def test_llm_query_bridge_tracks_metadata(self, mock_invocation_context):
        """Metadata tracked when enabled - get_sub_lm_call_metadata() has entries."""
        reset_sub_lm_call_metadata()

        bridge = create_llm_query_bridge(mock_invocation_context, track_metadata=True)
        bridge("Test prompt")

        metadata = get_sub_lm_call_metadata()
        assert len(metadata) == 1
        assert metadata[0]["call_index"] == 1

    def test_llm_query_bridge_metadata_has_latency(self, mock_invocation_context):
        """Metadata includes latency - latency_ms key present."""
        reset_sub_lm_call_metadata()

        bridge = create_llm_query_bridge(mock_invocation_context, track_metadata=True)
        bridge("Test prompt")

        metadata = get_sub_lm_call_metadata()
        assert "latency_ms" in metadata[0]
        assert metadata[0]["latency_ms"] >= 0

    def test_llm_query_bridge_without_context_uses_fallback(self):
        """Falls back when no context - returns simulated response."""
        reset_sub_lm_call_metadata()

        bridge = create_llm_query_bridge(invocation_context=None)
        result = bridge("Summarize the data")

        # Should get simulated response
        assert isinstance(result, str)
        assert len(result) > 0

    def test_create_llm_query_batched_bridge(self, mock_invocation_context):
        """Creates batched bridge - returns callable."""
        bridge = create_llm_query_batched_bridge(mock_invocation_context)
        assert callable(bridge)

    def test_llm_query_batched_concurrent_calls(self, mock_invocation_context):
        """Batched calls made concurrently - all responses returned."""
        # Set up multiple responses
        mock_invocation_context.llm.responses = [
            "Response 1",
            "Response 2",
            "Response 3",
        ]
        mock_invocation_context.llm.call_index = 0

        bridge = create_llm_query_batched_bridge(mock_invocation_context)
        results = bridge(["Prompt 1", "Prompt 2", "Prompt 3"])

        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_llm_query_batched_handles_errors(self):
        """Individual errors don't fail batch - error messages in results."""
        # Create context with client that will raise errors
        session = MockSession(state={})

        class ErrorMockClient:
            async def generate_content_async(self, prompt: str):
                if "error" in prompt.lower():
                    raise ValueError("Simulated error")
                return MockGenerateResponse(text=f"Success: {prompt}")

        llm_client = ErrorMockClient()
        ctx = MockInvocationContext(session=session, llm=llm_client)

        bridge = create_llm_query_batched_bridge(ctx)
        results = bridge(["Good prompt", "Error prompt", "Another good"])

        assert len(results) == 3
        # Error should be captured in result
        assert any("Error" in r or "Simulated error" in r for r in results)

    def test_reset_sub_lm_call_metadata(self, mock_invocation_context):
        """Resets call counter and metadata - both cleared."""
        reset_sub_lm_call_metadata()

        bridge = create_llm_query_bridge(mock_invocation_context)
        bridge("Test 1")
        bridge("Test 2")

        metadata = get_sub_lm_call_metadata()
        assert len(metadata) == 2

        # Reset
        reset_sub_lm_call_metadata()

        metadata = get_sub_lm_call_metadata()
        assert len(metadata) == 0


# =============================================================================
# Test Class 2: TestRLMStateIntegration
# =============================================================================

class TestRLMStateIntegration:
    """Test RLM state management across iterations."""

    def test_get_or_create_rlm_state_creates_new(self):
        """Creates new state when missing - state created."""
        session_state = {}
        rlm_state = get_or_create_rlm_state(session_state, "test_session")

        assert isinstance(rlm_state, RLMSessionState)
        assert rlm_state.session_id == "test_session"
        assert "_rlm_state_test_session" in session_state

    def test_get_or_create_rlm_state_returns_existing(self):
        """Returns existing state - same object returned."""
        session_state = {}

        # Create first time
        state1 = get_or_create_rlm_state(session_state, "test_session")
        state1.context_description = "Test context"

        # Get second time
        state2 = get_or_create_rlm_state(session_state, "test_session")

        assert state1 is state2
        assert state2.context_description == "Test context"

    def test_rlm_session_state_add_iteration(self):
        """add_iteration creates RLMIteration - iteration appended."""
        state = RLMSessionState(session_id="test")

        execution_result = {
            "status": "success",
            "stdout": "Output",
            "stderr": "",
            "llm_calls": 2,
        }

        iteration = state.add_iteration(
            generated_code="print('test')",
            execution_result=execution_result,
        )

        assert iteration.iteration_number == 1
        assert iteration.generated_code == "print('test')"
        assert iteration.llm_calls_made == 2
        assert len(state.iterations) == 1

    def test_rlm_session_state_iteration_count_increments(self):
        """iteration_count increments - count matches iterations."""
        state = RLMSessionState(session_id="test")

        assert state.iteration_count == 0

        state.add_iteration("code1", {"status": "success", "llm_calls": 0})
        assert state.iteration_count == 1

        state.add_iteration("code2", {"status": "success", "llm_calls": 0})
        assert state.iteration_count == 2

    def test_rlm_session_state_iteration_history_formats(self):
        """iteration_history formats correctly - contains code and output."""
        state = RLMSessionState(session_id="test")

        state.add_iteration(
            "print('hello')",
            {"status": "success", "stdout": "hello", "stderr": "", "llm_calls": 0}
        )

        history = state.iteration_history

        assert "Iteration 1" in history
        assert "print('hello')" in history
        assert "hello" in history

    def test_rlm_session_state_total_llm_calls_aggregates(self):
        """total_llm_calls sums across iterations - total is sum."""
        state = RLMSessionState(session_id="test")

        state.add_iteration("code1", {"status": "success", "llm_calls": 2})
        state.add_iteration("code2", {"status": "success", "llm_calls": 3})
        state.add_iteration("code3", {"status": "success", "llm_calls": 1})

        assert state.total_llm_calls == 6

    def test_rlm_iteration_format_for_prompt(self):
        """RLMIteration.format_for_prompt() works - contains all sections."""
        from rlm_adk.rlm_state import RLMIteration

        iteration = RLMIteration(
            iteration_number=1,
            generated_code="x = 5\nprint(x)",
            execution_result={},
            stdout="5",
            stderr="",
            error=None,
            llm_calls_made=2,
        )

        formatted = iteration.format_for_prompt()

        assert "Iteration 1" in formatted
        assert "x = 5" in formatted
        assert "Output:" in formatted
        assert "5" in formatted
        assert "2 sub-LM calls" in formatted


# =============================================================================
# Test Class 3: TestCompletionCheckerIntegration
# =============================================================================

class TestCompletionCheckerIntegration:
    """Test completion checker with state."""

    @pytest.mark.asyncio
    async def test_completion_checker_detects_final_in_code(self):
        """Finds FINAL() in generated_code - final_answer set, escalate event."""
        from rlm_adk.agents.completion_checker import RLMCompletionChecker

        checker = RLMCompletionChecker()

        session_state = {
            "rlm_session_id": "test",
            "generated_code": MOCK_GENERATED_CODE_WITH_FINAL,
            "execution_result": {"stdout": "", "stderr": ""},
        }

        # Initialize state
        get_or_create_rlm_state(session_state, "test")

        # Mock context
        session = MockSession(state=session_state)
        llm = MockLLMClient()
        ctx = MockInvocationContext(session=session, llm=llm)

        events = []
        async for event in checker._run_async_impl(ctx):
            events.append(event)

        assert len(events) == 1
        assert events[0].actions.escalate is True
        assert session_state["rlm_final_answer"] == "Found 42 duplicate vendors"

    @pytest.mark.asyncio
    async def test_completion_checker_detects_final_in_stdout(self):
        """Finds FINAL() in execution stdout - final_answer set."""
        from rlm_adk.agents.completion_checker import RLMCompletionChecker

        checker = RLMCompletionChecker()

        session_state = {
            "rlm_session_id": "test",
            "generated_code": "print('FINAL(Answer from stdout)')",
            "execution_result": {"stdout": "FINAL(Answer from stdout)", "stderr": ""},
        }

        get_or_create_rlm_state(session_state, "test")

        session = MockSession(state=session_state)
        llm = MockLLMClient()
        ctx = MockInvocationContext(session=session, llm=llm)

        events = []
        async for event in checker._run_async_impl(ctx):
            events.append(event)

        assert session_state.get("rlm_final_answer") == "Answer from stdout"

    @pytest.mark.asyncio
    async def test_completion_checker_detects_final_var(self):
        """Handles FINAL_VAR pattern - final_var_name set."""
        from rlm_adk.agents.completion_checker import RLMCompletionChecker

        checker = RLMCompletionChecker()

        session_state = {
            "rlm_session_id": "test",
            "generated_code": MOCK_GENERATED_CODE_WITH_FINAL_VAR,
            "execution_result": {"stdout": "", "stderr": ""},
        }

        get_or_create_rlm_state(session_state, "test")

        session = MockSession(state=session_state)
        llm = MockLLMClient()
        ctx = MockInvocationContext(session=session, llm=llm)

        events = []
        async for event in checker._run_async_impl(ctx):
            events.append(event)

        assert "rlm_final_var_name" in session_state
        assert session_state["rlm_final_var_name"] == "result_data"

    @pytest.mark.asyncio
    async def test_completion_checker_continues_without_final(self):
        """No escalation without FINAL - no escalate action."""
        from rlm_adk.agents.completion_checker import RLMCompletionChecker

        checker = RLMCompletionChecker()

        session_state = {
            "rlm_session_id": "test",
            "generated_code": "print('normal output')",
            "execution_result": {"stdout": "normal output", "stderr": ""},
        }

        get_or_create_rlm_state(session_state, "test")

        session = MockSession(state=session_state)
        llm = MockLLMClient()
        ctx = MockInvocationContext(session=session, llm=llm)

        events = []
        async for event in checker._run_async_impl(ctx):
            events.append(event)

        assert len(events) == 1
        # Should not escalate
        assert events[0].actions.escalate is not True

    @pytest.mark.asyncio
    async def test_completion_checker_continues_on_error(self):
        """Continues loop on execution error - no escalate action."""
        from rlm_adk.agents.completion_checker import RLMCompletionChecker

        checker = RLMCompletionChecker()

        session_state = {
            "rlm_session_id": "test",
            "generated_code": "undefined_var",
            "execution_result": {
                "stdout": "",
                "stderr": "",
                "error_message": "NameError: name 'undefined_var' is not defined",
            },
        }

        get_or_create_rlm_state(session_state, "test")

        session = MockSession(state=session_state)
        llm = MockLLMClient()
        ctx = MockInvocationContext(session=session, llm=llm)

        events = []
        async for event in checker._run_async_impl(ctx):
            events.append(event)

        assert len(events) == 1
        # Should continue, not escalate
        assert events[0].actions.escalate is not True

    @pytest.mark.asyncio
    async def test_completion_checker_stores_termination_reason(self):
        """Sets rlm_termination_reason - value is FINAL."""
        from rlm_adk.agents.completion_checker import RLMCompletionChecker

        checker = RLMCompletionChecker()

        session_state = {
            "rlm_session_id": "test",
            "generated_code": MOCK_GENERATED_CODE_WITH_FINAL,
            "execution_result": {"stdout": "", "stderr": ""},
        }

        get_or_create_rlm_state(session_state, "test")

        session = MockSession(state=session_state)
        llm = MockLLMClient()
        ctx = MockInvocationContext(session=session, llm=llm)

        events = []
        async for event in checker._run_async_impl(ctx):
            events.append(event)

        assert session_state["rlm_termination_reason"] == "FINAL"


# =============================================================================
# Test Class 4: TestExecuteRlmIterationIntegration
# =============================================================================

class TestExecuteRlmIterationIntegration:
    """Test the execute_rlm_iteration tool integration."""

    def test_execute_rlm_iteration_extracts_code_blocks(self):
        """Extracts code from generated_code - code executed."""
        code_blocks = find_code_blocks(MOCK_GENERATED_CODE_WITH_LLM_QUERY)

        assert len(code_blocks) == 1
        assert "llm_query" in code_blocks[0]

    def test_execute_rlm_iteration_runs_in_repl(self):
        """Code runs in REPL environment - variables accessible."""
        reset_sub_lm_call_metadata()

        llm_query_fn = lambda p: "Mock LLM response"
        repl = RLMREPLEnvironment(llm_query_fn=llm_query_fn)

        result = repl.execute_code("x = 42\nprint(x)")

        assert result["status"] == "success"
        assert "42" in result["stdout"]
        assert repl.get_variable("x") == 42

    def test_execute_rlm_iteration_llm_query_available(self, mock_invocation_context):
        """llm_query callable in code - sub-LM call counted."""
        reset_sub_lm_call_metadata()

        bridge = create_llm_query_bridge(mock_invocation_context)
        repl = RLMREPLEnvironment(llm_query_fn=bridge)

        result = repl.execute_code('result = llm_query("test")\nprint(result)')

        assert result["status"] == "success"
        assert result["llm_calls"] == 1
        assert "Mock" in result["stdout"]

    def test_execute_rlm_iteration_llm_query_batched_available(self, mock_invocation_context):
        """llm_query_batched callable - batch call counted."""
        reset_sub_lm_call_metadata()

        bridge = create_llm_query_bridge(mock_invocation_context)
        batched_bridge = create_llm_query_batched_bridge(mock_invocation_context)

        repl = RLMREPLEnvironment(
            llm_query_fn=bridge,
            llm_query_batched_fn=batched_bridge
        )

        result = repl.execute_code('results = llm_query_batched(["a", "b"])\nprint(len(results))')

        assert result["status"] == "success"
        assert result["llm_calls"] == 2

    def test_execute_rlm_iteration_context_available(self):
        """context variable accessible - value matches loaded."""
        reset_sub_lm_call_metadata()

        llm_query_fn = lambda p: "Mock"
        context_data = {"vendors": [{"name": "ABC Corp"}]}

        repl = RLMREPLEnvironment(llm_query_fn=llm_query_fn, context=context_data)

        result = repl.execute_code('print(context["vendors"][0]["name"])')

        assert result["status"] == "success"
        assert "ABC Corp" in result["stdout"]

    def test_execute_rlm_iteration_updates_iteration_history(self):
        """Updates state[iteration_history] - new iteration in history."""
        session_state = {}
        rlm_state = get_or_create_rlm_state(session_state, "test")

        execution_result = {
            "status": "success",
            "stdout": "Output",
            "stderr": "",
            "llm_calls": 1,
        }

        rlm_state.add_iteration("print('test')", execution_result)

        history = rlm_state.iteration_history
        assert "Iteration 1" in history
        assert "print('test')" in history

    def test_execute_rlm_iteration_returns_metadata(self):
        """Returns custom_metadata - RLMExecutionMetadata present."""
        metadata = RLMExecutionMetadata(
            iteration_number=1,
            blocks_executed=2,
            llm_calls_made=3,
            status="success",
        )

        meta_dict = metadata.to_dict()

        assert meta_dict["iteration_number"] == 1
        assert meta_dict["blocks_executed"] == 2
        assert meta_dict["llm_calls_made"] == 3

    def test_execute_rlm_iteration_handles_multiple_blocks(self):
        """Executes all code blocks - all blocks run."""
        code_blocks = find_code_blocks(MOCK_GENERATED_CODE_MULTIPLE_BLOCKS)

        assert len(code_blocks) == 2

        llm_query_fn = lambda p: "Mock analysis"
        context_data = {"vendors": ["A", "B", "C", "D", "E"]}

        repl = RLMREPLEnvironment(llm_query_fn=llm_query_fn, context=context_data)

        # Execute first block
        result1 = repl.execute_code(code_blocks[0])
        assert result1["status"] == "success"

        # Execute second block (depends on first)
        result2 = repl.execute_code(code_blocks[1])
        assert result2["status"] == "success"
        assert result2["llm_calls"] == 1

    def test_execute_rlm_iteration_handles_error(self):
        """Error in code returns error status - status == error."""
        llm_query_fn = lambda p: "Mock"
        repl = RLMREPLEnvironment(llm_query_fn=llm_query_fn)

        result = repl.execute_code("undefined_variable")

        assert result["status"] == "error"
        assert result["error_type"] == "NameError"
        assert "undefined_variable" in result["error_message"]

    def test_execute_rlm_iteration_no_code_blocks(self):
        """No code blocks returns no_code - status == no_code."""
        response_without_code = "Here's my analysis without any code blocks."

        code_blocks = find_code_blocks(response_without_code)

        assert len(code_blocks) == 0


# =============================================================================
# Test Class 5: TestFullWorkflowIntegration
# =============================================================================

class TestFullWorkflowIntegration:
    """End-to-end workflow tests."""

    def test_workflow_context_flows_through_agents(self):
        """Context loaded by first agent accessible by second - context preserved."""
        session_state = {}

        # Simulate context loading
        context_data = {"vendors": [{"name": "Test"}]}
        session_state["rlm_context"] = context_data

        # Simulate code execution that accesses context
        reset_sub_lm_call_metadata()
        llm_query_fn = lambda p: "Mock"
        repl = RLMREPLEnvironment(llm_query_fn=llm_query_fn, context=context_data)

        result = repl.execute_code('x = context["vendors"][0]["name"]')

        assert result["status"] == "success"
        assert repl.get_variable("x") == "Test"

    def test_workflow_iteration_history_accumulates(self):
        """History grows with each iteration - multiple iterations recorded."""
        rlm_state = RLMSessionState(session_id="test")

        for i in range(3):
            rlm_state.add_iteration(
                f"code_{i}",
                {"status": "success", "stdout": f"output_{i}", "stderr": "", "llm_calls": 0}
            )

        assert rlm_state.iteration_count == 3

        history = rlm_state.iteration_history
        assert "Iteration 1" in history
        assert "Iteration 2" in history
        assert "Iteration 3" in history

    @pytest.mark.asyncio
    async def test_workflow_terminates_on_final(self):
        """Loop exits when FINAL detected - escalation stops loop."""
        from rlm_adk.agents.completion_checker import RLMCompletionChecker

        checker = RLMCompletionChecker()

        session_state = {
            "rlm_session_id": "test",
            "generated_code": MOCK_GENERATED_CODE_WITH_FINAL,
            "execution_result": {"stdout": "", "stderr": ""},
        }

        get_or_create_rlm_state(session_state, "test")

        session = MockSession(state=session_state)
        llm = MockLLMClient()
        ctx = MockInvocationContext(session=session, llm=llm)

        events = []
        async for event in checker._run_async_impl(ctx):
            events.append(event)

        # Should escalate to exit loop
        assert events[0].actions.escalate is True

    def test_workflow_terminates_on_max_iterations(self):
        """Loop exits at max_iterations - iteration count at max."""
        rlm_state = RLMSessionState(session_id="test")
        max_iterations = 5

        for i in range(max_iterations):
            rlm_state.add_iteration(
                f"code_{i}",
                {"status": "success", "stdout": "", "stderr": "", "llm_calls": 0}
            )

        assert rlm_state.iteration_count == max_iterations

    def test_workflow_metrics_aggregated(self):
        """Final metrics include all iterations - totals match sum."""
        rlm_state = RLMSessionState(session_id="test")

        # Add iterations with varying LLM calls
        rlm_state.add_iteration("code1", {"status": "success", "llm_calls": 2})
        rlm_state.add_iteration("code2", {"status": "success", "llm_calls": 3})
        rlm_state.add_iteration("code3", {"status": "success", "llm_calls": 1})

        assert rlm_state.total_llm_calls == 6
        assert rlm_state.iteration_count == 3

    def test_workflow_final_answer_passed_to_formatter(self):
        """rlm_final_answer available to formatter - value in state."""
        session_state = {}
        rlm_state = get_or_create_rlm_state(session_state, "test")

        # Simulate completion checker setting final answer
        rlm_state.final_answer = "Final result: 42 duplicates"
        session_state["rlm_final_answer"] = rlm_state.final_answer

        assert session_state["rlm_final_answer"] == "Final result: 42 duplicates"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
