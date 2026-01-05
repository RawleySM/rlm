"""Tests for RLM REPL and RLM-ADK integration.

Tests the core RLM functionality including:
- RLMREPLEnvironment class with llm_query and context
- RLM tools (rlm_execute_code, rlm_load_context, etc.)
- Context loaders for Unity Catalog data

These tests do NOT require google-adk to be installed.
"""

import pytest

from rlm_adk.testing import create_tool_context
from rlm_adk.rlm_repl import (
    RLMREPLEnvironment,
    find_code_blocks,
    find_final_answer,
    get_or_create_repl_session,
    clear_repl_session,
)


@pytest.fixture(autouse=True)
def clear_databricks_env(monkeypatch):
    """Clear Databricks environment variables to force simulation mode."""
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.delenv("DATABRICKS_CLUSTER_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_SQL_WAREHOUSE_ID", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture
def mock_llm_query():
    """Create a mock llm_query function for testing."""
    call_count = [0]

    def llm_query(prompt: str) -> str:
        call_count[0] += 1
        if "duplicate" in prompt.lower():
            return "Found 3 potential duplicates based on tax ID matching."
        elif "analyze" in prompt.lower():
            return "Analysis complete: 5 vendors identified."
        elif "summarize" in prompt.lower():
            return "Summary: Data contains vendor records from multiple sources."
        else:
            return f"Response to: {prompt[:50]}..."

    llm_query.call_count = call_count
    return llm_query


@pytest.fixture
def repl_env(mock_llm_query):
    """Create an RLM REPL environment for testing."""
    return RLMREPLEnvironment(
        llm_query_fn=mock_llm_query,
        context={"test_data": [1, 2, 3]},
    )


class TestRLMREPLEnvironment:
    """Tests for RLMREPLEnvironment class."""

    def test_init_with_context(self, mock_llm_query):
        """Test REPL environment initializes with context."""
        env = RLMREPLEnvironment(
            llm_query_fn=mock_llm_query,
            context={"vendors": ["A", "B", "C"]},
        )
        assert env.context == {"vendors": ["A", "B", "C"]}

    def test_execute_simple_code(self, repl_env):
        """Test executing simple Python code."""
        result = repl_env.execute_code("x = 1 + 1")
        assert result["status"] == "success"
        assert repl_env.get_variable("x") == 2

    def test_execute_code_with_print(self, repl_env):
        """Test code that produces stdout."""
        result = repl_env.execute_code("print('hello')")
        assert result["status"] == "success"
        assert "hello" in result["stdout"]

    def test_execute_code_with_context_access(self, repl_env):
        """Test code can access the context variable."""
        result = repl_env.execute_code("data = context['test_data']")
        assert result["status"] == "success"
        assert repl_env.get_variable("data") == [1, 2, 3]

    def test_execute_code_with_llm_query(self, repl_env, mock_llm_query):
        """Test code can call llm_query for recursive decomposition."""
        code = """
result = llm_query("Analyze this data for duplicates")
print(f"LLM said: {result}")
"""
        result = repl_env.execute_code(code)
        assert result["status"] == "success"
        assert result["llm_calls"] == 1
        assert "duplicates" in result["stdout"].lower()

    def test_execute_code_with_llm_query_batched(self, repl_env):
        """Test code can call llm_query_batched for concurrent calls."""
        code = """
prompts = ["Analyze chunk 1", "Analyze chunk 2"]
results = llm_query_batched(prompts)
print(f"Got {len(results)} responses")
"""
        result = repl_env.execute_code(code)
        assert result["status"] == "success"
        assert result["llm_calls"] == 2
        assert "2 responses" in result["stdout"]

    def test_execute_code_error_handling(self, repl_env):
        """Test error handling in code execution."""
        result = repl_env.execute_code("raise ValueError('test error')")
        assert result["status"] == "error"
        assert result["error_type"] == "ValueError"
        assert "test error" in result["error_message"]

    def test_variable_persistence(self, repl_env):
        """Test variables persist across executions."""
        repl_env.execute_code("x = 10")
        repl_env.execute_code("y = x * 2")
        assert repl_env.get_variable("y") == 20

    def test_context_modification(self, repl_env):
        """Test context can be modified."""
        repl_env.context = {"new": "data"}
        result = repl_env.execute_code("value = context['new']")
        assert result["status"] == "success"
        assert repl_env.get_variable("value") == "data"

    def test_get_stats(self, repl_env, mock_llm_query):
        """Test execution statistics tracking."""
        repl_env.execute_code("x = 1")
        repl_env.execute_code("y = llm_query('test')")

        stats = repl_env.get_stats()
        assert stats["execution_count"] == 2
        assert stats["llm_call_count"] == 1
        assert "x" in stats["variables_defined"]
        assert "y" in stats["variables_defined"]

    def test_reset(self, repl_env):
        """Test environment reset."""
        repl_env.execute_code("x = 1")
        repl_env.reset()

        assert repl_env.get_variable("x") is None
        stats = repl_env.get_stats()
        assert stats["execution_count"] == 0
        assert stats["llm_call_count"] == 0


class TestCodeBlockParsing:
    """Tests for code block extraction from LM responses."""

    def test_find_python_code_block(self):
        """Test extracting Python code blocks."""
        response = """
Here's my analysis:

```python
x = 1 + 1
print(x)
```

That should work.
"""
        blocks = find_code_blocks(response)
        assert len(blocks) == 1
        assert "x = 1 + 1" in blocks[0]

    def test_find_repl_code_block(self):
        """Test extracting REPL code blocks."""
        response = """
```repl
result = llm_query("analyze this")
print(result)
```
"""
        blocks = find_code_blocks(response)
        assert len(blocks) == 1
        assert "llm_query" in blocks[0]

    def test_find_multiple_code_blocks(self):
        """Test extracting multiple code blocks."""
        response = """
```python
step1 = "first"
```

And then:

```python
step2 = "second"
```
"""
        blocks = find_code_blocks(response)
        assert len(blocks) == 2

    def test_no_code_blocks(self):
        """Test response with no code blocks."""
        response = "Just plain text without any code."
        blocks = find_code_blocks(response)
        assert len(blocks) == 0


class TestFinalAnswerParsing:
    """Tests for FINAL answer extraction."""

    def test_find_final_answer(self):
        """Test extracting FINAL(answer) pattern."""
        response = "After analysis, FINAL(42 duplicates found)"
        answer = find_final_answer(response)
        assert answer == "42 duplicates found"

    def test_find_final_var(self):
        """Test extracting FINAL_VAR(variable) pattern."""
        response = "The result is stored in FINAL_VAR(result_data)"
        answer = find_final_answer(response)
        assert answer == "__FINAL_VAR__:result_data"

    def test_no_final_answer(self):
        """Test response without FINAL pattern."""
        response = "Still processing, not done yet."
        answer = find_final_answer(response)
        assert answer is None


class TestSessionManagement:
    """Tests for REPL session management."""

    def test_get_or_create_session(self, mock_llm_query):
        """Test creating a new session."""
        session_id = "test_session_123"
        clear_repl_session(session_id)  # Ensure clean state

        session = get_or_create_repl_session(
            session_id=session_id,
            llm_query_fn=mock_llm_query,
            context={"test": True},
        )

        assert session.context == {"test": True}

        # Getting same session should return existing
        session2 = get_or_create_repl_session(
            session_id=session_id,
            llm_query_fn=mock_llm_query,
        )
        assert session is session2

    def test_clear_session(self, mock_llm_query):
        """Test clearing a session."""
        session_id = "test_session_clear"

        session1 = get_or_create_repl_session(
            session_id=session_id,
            llm_query_fn=mock_llm_query,
        )
        session1.execute_code("x = 100")

        clear_repl_session(session_id)

        session2 = get_or_create_repl_session(
            session_id=session_id,
            llm_query_fn=mock_llm_query,
        )

        # Should be a new session without the variable
        assert session2.get_variable("x") is None


class TestRLMTools:
    """Tests for RLM-ADK tools."""

    def test_rlm_execute_code(self):
        """Test rlm_execute_code tool."""
        from rlm_adk.tools.rlm_tools import rlm_execute_code

        ctx = create_tool_context({"rlm_session_id": "test_exec"})
        result = rlm_execute_code("x = 2 + 2", ctx)

        assert result["status"] == "success"

    def test_rlm_execute_code_with_context(self):
        """Test rlm_execute_code can access loaded context."""
        from rlm_adk.tools.rlm_tools import rlm_execute_code, rlm_load_context

        ctx = create_tool_context({"rlm_session_id": "test_exec_ctx"})

        # Load context first
        rlm_load_context(
            context_data={"vendors": [1, 2, 3]},
            context_description="Test vendors",
            tool_context=ctx,
        )

        # Execute code that uses context
        result = rlm_execute_code("vendor_count = len(context['vendors'])", ctx)

        assert result["status"] == "success"

    def test_rlm_load_context(self):
        """Test rlm_load_context tool."""
        from rlm_adk.tools.rlm_tools import rlm_load_context

        ctx = create_tool_context({"rlm_session_id": "test_load"})
        result = rlm_load_context(
            context_data=[1, 2, 3, 4, 5],
            context_description="Test numbers",
            tool_context=ctx,
        )

        assert result["status"] == "success"
        assert result["context_type"] == "list"
        assert result["context_size"] == 5
        assert "Test numbers" in result["description"]

    def test_rlm_query_context_no_context(self):
        """Test rlm_query_context without loaded context."""
        from rlm_adk.tools.rlm_tools import rlm_query_context

        ctx = create_tool_context({"rlm_session_id": "test_query_empty"})
        result = rlm_query_context(
            query="Find duplicates",
            strategy="chunk_and_aggregate",
            tool_context=ctx,
        )

        assert result["status"] == "error"
        assert "No context loaded" in result["error_message"]

    def test_rlm_query_context_chunk_and_aggregate(self):
        """Test chunk_and_aggregate strategy."""
        from rlm_adk.tools.rlm_tools import rlm_load_context, rlm_query_context

        ctx = create_tool_context({"rlm_session_id": "test_chunk_agg"})

        # Load test data
        rlm_load_context(
            context_data={"alpha": [1, 2], "beta": [3, 4]},
            context_description="Test data",
            tool_context=ctx,
        )

        result = rlm_query_context(
            query="Analyze the data",
            strategy="chunk_and_aggregate",
            tool_context=ctx,
        )

        assert result["status"] == "success"
        assert result["strategy_used"] == "chunk_and_aggregate"
        assert result["llm_calls"] > 0

    def test_rlm_get_session_state(self):
        """Test rlm_get_session_state tool."""
        from rlm_adk.tools.rlm_tools import rlm_get_session_state, rlm_execute_code

        ctx = create_tool_context({"rlm_session_id": "test_state"})

        # Execute some code first
        rlm_execute_code("my_var = 42", ctx)

        result = rlm_get_session_state(ctx)

        assert result["status"] == "success"
        assert result["session_id"] == "test_state"
        assert result["execution_count"] >= 1
        assert "my_var" in result["variables"]

    def test_rlm_clear_session(self):
        """Test rlm_clear_session tool."""
        from rlm_adk.tools.rlm_tools import (
            rlm_clear_session,
            rlm_execute_code,
            rlm_get_session_state,
        )

        ctx = create_tool_context({"rlm_session_id": "test_clear"})

        rlm_execute_code("x = 1", ctx)
        rlm_clear_session(ctx)

        state = rlm_get_session_state(ctx)
        assert state["execution_count"] == 0


class TestContextLoaders:
    """Tests for context loader tools."""

    def test_load_vendor_data_to_context(self):
        """Test loading vendor data from hospital chains."""
        from rlm_adk.tools.context_loader import load_vendor_data_to_context

        ctx = create_tool_context({"rlm_session_id": "test_vendor_load"})
        result = load_vendor_data_to_context(
            hospital_chains=["hospital_chain_alpha", "hospital_chain_beta"],
            include_masterdata=True,
            tool_context=ctx,
        )

        assert result["status"] == "success"
        assert "hospital_chain_alpha" in result["chains_loaded"]
        assert "hospital_chain_beta" in result["chains_loaded"]
        assert result["masterdata_included"] is True
        assert result["total_vendors"] > 0

    def test_load_vendor_data_without_masterdata(self):
        """Test loading vendor data without masterdata."""
        from rlm_adk.tools.context_loader import load_vendor_data_to_context

        ctx = create_tool_context({"rlm_session_id": "test_vendor_no_md"})
        result = load_vendor_data_to_context(
            hospital_chains=["hospital_chain_gamma"],
            include_masterdata=False,
            tool_context=ctx,
        )

        assert result["status"] == "success"
        assert result["masterdata_included"] is False
        assert "masterdata" not in result.get("context_structure", {})

    def test_load_custom_context(self):
        """Test loading custom context data."""
        from rlm_adk.tools.context_loader import load_custom_context

        ctx = create_tool_context({"rlm_session_id": "test_custom"})
        result = load_custom_context(
            data={"custom_field": "custom_value", "numbers": [1, 2, 3]},
            description="Custom test data",
            tool_context=ctx,
        )

        assert result["status"] == "success"
        assert result["context_type"] == "dict"

    def test_load_query_results_to_context(self):
        """Test loading SQL query results into context."""
        from rlm_adk.tools.context_loader import load_query_results_to_context

        ctx = create_tool_context({"rlm_session_id": "test_query_load"})
        result = load_query_results_to_context(
            sql_query="SELECT * FROM vendors LIMIT 10",
            description="Vendor query results",
            tool_context=ctx,
        )

        assert result["status"] == "success"
        assert result["rows_loaded"] >= 0


class TestRLMDecompositionStrategies:
    """Tests for RLM decomposition strategies."""

    def test_iterative_strategy(self):
        """Test iterative decomposition strategy."""
        from rlm_adk.tools.rlm_tools import rlm_load_context, rlm_query_context

        ctx = create_tool_context({"rlm_session_id": "test_iterative"})

        rlm_load_context(
            context_data=[{"id": 1}, {"id": 2}, {"id": 3}],
            context_description="Test items",
            tool_context=ctx,
        )

        result = rlm_query_context(
            query="Process each item",
            strategy="iterative",
            tool_context=ctx,
        )

        assert result["status"] == "success"
        assert result["strategy_used"] == "iterative"

    def test_map_reduce_strategy(self):
        """Test map-reduce decomposition strategy."""
        from rlm_adk.tools.rlm_tools import rlm_load_context, rlm_query_context

        ctx = create_tool_context({"rlm_session_id": "test_map_reduce"})

        rlm_load_context(
            context_data=["item1", "item2", "item3"],
            context_description="Test items for map-reduce",
            tool_context=ctx,
        )

        result = rlm_query_context(
            query="Map and reduce items",
            strategy="map_reduce",
            tool_context=ctx,
        )

        assert result["status"] == "success"
        assert result["strategy_used"] == "map_reduce"

    def test_hierarchical_strategy(self):
        """Test hierarchical decomposition strategy."""
        from rlm_adk.tools.rlm_tools import rlm_load_context, rlm_query_context

        ctx = create_tool_context({"rlm_session_id": "test_hierarchical"})

        rlm_load_context(
            context_data={
                "level1_a": {"sub": "data_a"},
                "level1_b": {"sub": "data_b"},
            },
            context_description="Hierarchical test data",
            tool_context=ctx,
        )

        result = rlm_query_context(
            query="Build hierarchical summary",
            strategy="hierarchical",
            tool_context=ctx,
        )

        assert result["status"] == "success"
        assert result["strategy_used"] == "hierarchical"
