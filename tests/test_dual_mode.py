"""Tests for runtime detection and dual-mode tools.

Tests the dual-mode execution feature in rlm_adk which enables:
- Local Orchestration: REST APIs for remote Databricks execution
- Native Execution: Direct SparkSession on Databricks clusters

These tests do NOT require actual Databricks connection or PySpark installation.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestRuntimeDetection:
    """Tests for rlm_adk/runtime.py"""

    def setup_method(self):
        """Clear cache before each test."""
        from rlm_adk.runtime import clear_execution_mode_cache

        clear_execution_mode_cache()

    def test_default_mode_is_local(self):
        """Without any env vars, should default to local mode."""
        from rlm_adk.runtime import clear_execution_mode_cache, get_execution_mode

        with patch.dict(os.environ, {}, clear=True):
            clear_execution_mode_cache()
            # Remove Databricks markers
            with patch("os.path.exists", return_value=False):
                assert get_execution_mode() == "local"

    def test_explicit_override_native(self):
        """RLM_EXECUTION_MODE=native should force native mode."""
        from rlm_adk.runtime import clear_execution_mode_cache, get_execution_mode

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "native"}):
            clear_execution_mode_cache()
            assert get_execution_mode() == "native"

    def test_explicit_override_local(self):
        """RLM_EXECUTION_MODE=local should force local mode."""
        from rlm_adk.runtime import clear_execution_mode_cache, get_execution_mode

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}):
            clear_execution_mode_cache()
            assert get_execution_mode() == "local"

    def test_explicit_override_case_insensitive(self):
        """RLM_EXECUTION_MODE should be case insensitive."""
        from rlm_adk.runtime import clear_execution_mode_cache, get_execution_mode

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "NATIVE"}):
            clear_execution_mode_cache()
            assert get_execution_mode() == "native"

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "LOCAL"}):
            clear_execution_mode_cache()
            assert get_execution_mode() == "local"

    def test_invalid_mode_raises_error(self):
        """Invalid RLM_EXECUTION_MODE should raise ValueError."""
        from rlm_adk.runtime import clear_execution_mode_cache, get_execution_mode

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "invalid"}):
            clear_execution_mode_cache()
            with pytest.raises(ValueError, match="Invalid RLM_EXECUTION_MODE"):
                get_execution_mode()

    def test_auto_detect_databricks_runtime_version(self):
        """DATABRICKS_RUNTIME_VERSION should trigger native mode."""
        from rlm_adk.runtime import is_databricks_runtime

        with patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "14.3"}):
            assert is_databricks_runtime() is True

    def test_auto_detect_databricks_spark_path(self):
        """Presence of /databricks/spark should trigger native mode."""
        from rlm_adk.runtime import is_databricks_runtime

        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=True):
                assert is_databricks_runtime() is True

    def test_not_databricks_runtime_without_markers(self):
        """Without markers, is_databricks_runtime should return False."""
        from rlm_adk.runtime import is_databricks_runtime

        with patch.dict(os.environ, {}, clear=True):
            with patch("os.path.exists", return_value=False):
                assert is_databricks_runtime() is False

    def test_get_spark_session_returns_none_in_local(self):
        """get_spark_session() should return None in local mode."""
        from rlm_adk.runtime import clear_execution_mode_cache, get_spark_session

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}):
            clear_execution_mode_cache()
            assert get_spark_session() is None

    def test_get_dbutils_returns_none_in_local(self):
        """get_dbutils() should return None in local mode."""
        from rlm_adk.runtime import clear_execution_mode_cache, get_dbutils

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}):
            clear_execution_mode_cache()
            assert get_dbutils() is None

    def test_llm_timeout_default(self):
        """Default LLM timeout should be 60."""
        from rlm_adk.runtime import get_llm_timeout_seconds

        with patch.dict(os.environ, {}, clear=True):
            assert get_llm_timeout_seconds() == 60

    def test_llm_timeout_custom(self):
        """Custom LLM timeout from env var."""
        from rlm_adk.runtime import get_llm_timeout_seconds

        with patch.dict(os.environ, {"RLM_LLM_TIMEOUT_SECONDS": "120"}):
            assert get_llm_timeout_seconds() == 120

    def test_llm_timeout_invalid_falls_back_to_default(self):
        """Invalid LLM timeout should fall back to default."""
        from rlm_adk.runtime import get_llm_timeout_seconds

        with patch.dict(os.environ, {"RLM_LLM_TIMEOUT_SECONDS": "not_a_number"}):
            assert get_llm_timeout_seconds() == 60

    def test_llm_max_retries_default(self):
        """Default LLM max retries should be 3."""
        from rlm_adk.runtime import get_llm_max_retries

        with patch.dict(os.environ, {}, clear=True):
            assert get_llm_max_retries() == 3

    def test_llm_max_retries_custom(self):
        """Custom LLM max retries from env var."""
        from rlm_adk.runtime import get_llm_max_retries

        with patch.dict(os.environ, {"RLM_LLM_MAX_RETRIES": "5"}):
            assert get_llm_max_retries() == 5

    def test_llm_max_retries_invalid_falls_back_to_default(self):
        """Invalid LLM max retries should fall back to default."""
        from rlm_adk.runtime import get_llm_max_retries

        with patch.dict(os.environ, {"RLM_LLM_MAX_RETRIES": "invalid"}):
            assert get_llm_max_retries() == 3

    def test_cache_is_effective(self):
        """Execution mode should be cached after first call."""
        from rlm_adk.runtime import clear_execution_mode_cache, get_execution_mode

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}):
            clear_execution_mode_cache()
            first_call = get_execution_mode()

            # Change the env var
            os.environ["RLM_EXECUTION_MODE"] = "native"

            # Should still return local (cached)
            second_call = get_execution_mode()
            assert first_call == second_call == "local"

    def test_get_runtime_info(self):
        """get_runtime_info should return comprehensive info dict."""
        from rlm_adk.runtime import clear_execution_mode_cache, get_runtime_info

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}, clear=True):
            clear_execution_mode_cache()
            info = get_runtime_info()

            assert "execution_mode" in info
            assert "is_databricks_runtime" in info
            assert "databricks_runtime_version" in info
            assert "explicit_mode_override" in info
            assert "spark_available" in info
            assert "dbutils_available" in info
            assert "llm_timeout_seconds" in info
            assert "llm_max_retries" in info
            assert info["execution_mode"] == "local"


class TestDualModeTools:
    """Tests for dual-mode tool dispatch."""

    def setup_method(self):
        from rlm_adk.runtime import clear_execution_mode_cache

        clear_execution_mode_cache()

    def test_execute_sql_dispatches_to_simulation_in_local(self):
        """In local mode without credentials, should use simulation."""
        from rlm_adk.runtime import clear_execution_mode_cache
        from rlm_adk.testing import create_tool_context
        from rlm_adk.tools.databricks_repl import execute_sql_query

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}, clear=True):
            clear_execution_mode_cache()

            mock_context = create_tool_context()

            result = execute_sql_query("SHOW CATALOGS", mock_context)
            assert result["status"] == "success"
            assert "catalogs" in str(result) or "data" in result

    def test_execute_python_dispatches_to_simulation_in_local(self):
        """In local mode without credentials, should use simulation."""
        from rlm_adk.runtime import clear_execution_mode_cache
        from rlm_adk.testing import create_tool_context
        from rlm_adk.tools.databricks_repl import execute_python_code

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}, clear=True):
            clear_execution_mode_cache()

            mock_context = create_tool_context({"repl_session_id": "test_session"})

            result = execute_python_code("x = 1 + 1", mock_context)
            assert result["status"] == "success"

    def test_list_catalogs_returns_success_in_local(self):
        """list_catalogs should work in local simulation mode."""
        from rlm_adk.runtime import clear_execution_mode_cache
        from rlm_adk.testing import create_tool_context
        from rlm_adk.tools.unity_catalog import list_catalogs

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}, clear=True):
            clear_execution_mode_cache()

            mock_context = create_tool_context()
            result = list_catalogs(mock_context)
            assert result["status"] == "success"
            assert "catalogs" in result

    def test_list_schemas_returns_success_in_local(self):
        """list_schemas should work in local simulation mode."""
        from rlm_adk.runtime import clear_execution_mode_cache
        from rlm_adk.testing import create_tool_context
        from rlm_adk.tools.unity_catalog import list_schemas

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}, clear=True):
            clear_execution_mode_cache()

            mock_context = create_tool_context()
            result = list_schemas("hospital_chain_alpha", mock_context)
            assert result["status"] == "success"
            assert "schemas" in result

    def test_list_tables_returns_success_in_local(self):
        """list_tables should work in local simulation mode."""
        from rlm_adk.runtime import clear_execution_mode_cache
        from rlm_adk.testing import create_tool_context
        from rlm_adk.tools.unity_catalog import list_tables

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}, clear=True):
            clear_execution_mode_cache()

            mock_context = create_tool_context()
            result = list_tables("hospital_chain_alpha", "erp_vendors", mock_context)
            assert result["status"] == "success"
            assert "tables" in result

    def test_sql_select_query_in_local_mode(self):
        """SQL SELECT queries should return simulated data in local mode."""
        from rlm_adk.runtime import clear_execution_mode_cache
        from rlm_adk.testing import create_tool_context
        from rlm_adk.tools.databricks_repl import execute_sql_query

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}, clear=True):
            clear_execution_mode_cache()

            mock_context = create_tool_context()
            result = execute_sql_query("SELECT * FROM vendors LIMIT 10", mock_context)

            assert result["status"] == "success"
            assert "columns" in result
            assert "data" in result

    def test_sql_describe_query_in_local_mode(self):
        """SQL DESCRIBE queries should return simulated schema in local mode."""
        from rlm_adk.runtime import clear_execution_mode_cache
        from rlm_adk.testing import create_tool_context
        from rlm_adk.tools.databricks_repl import execute_sql_query

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}, clear=True):
            clear_execution_mode_cache()

            mock_context = create_tool_context()
            result = execute_sql_query("DESCRIBE vendors", mock_context)

            assert result["status"] == "success"
            assert "columns" in result

    def test_sql_create_view_in_local_mode(self):
        """SQL CREATE VIEW should return success in local mode."""
        from rlm_adk.runtime import clear_execution_mode_cache
        from rlm_adk.testing import create_tool_context
        from rlm_adk.tools.databricks_repl import execute_sql_query

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}, clear=True):
            clear_execution_mode_cache()

            mock_context = create_tool_context()
            result = execute_sql_query(
                "CREATE VIEW test_view AS SELECT * FROM vendors", mock_context
            )

            assert result["status"] == "success"


class TestRLMREPLSparkInjection:
    """Tests for Spark injection in RLM REPL."""

    def setup_method(self):
        from rlm_adk.runtime import clear_execution_mode_cache

        clear_execution_mode_cache()

    def test_repl_without_spark_in_local_mode(self):
        """REPL should have spark=None in local mode."""
        from rlm_adk.rlm_repl import RLMREPLEnvironment
        from rlm_adk.runtime import clear_execution_mode_cache

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}):
            clear_execution_mode_cache()

            repl = RLMREPLEnvironment(
                llm_query_fn=lambda x: "test response",
                context={"test": "data"},
            )
            assert repl._spark is None

    def test_spark_available_in_exec_namespace(self):
        """spark should be in exec namespace even if None."""
        from rlm_adk.rlm_repl import RLMREPLEnvironment
        from rlm_adk.runtime import clear_execution_mode_cache

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}):
            clear_execution_mode_cache()

            repl = RLMREPLEnvironment(
                llm_query_fn=lambda x: "test response",
            )

            # Execute code that checks for spark
            result = repl.execute_code("spark_available = spark is not None")
            assert result["status"] == "success"
            assert repl.get_variable("spark_available") is False

    def test_context_lazy_resolution_without_spark(self):
        """String context should remain string without Spark."""
        from rlm_adk.rlm_repl import RLMREPLEnvironment
        from rlm_adk.runtime import clear_execution_mode_cache

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}):
            clear_execution_mode_cache()

            repl = RLMREPLEnvironment(
                llm_query_fn=lambda x: "test",
                context="catalog.schema.table",
            )
            # Should return string as-is since no Spark
            assert repl.context == "catalog.schema.table"

    def test_repl_with_explicit_spark_parameter(self):
        """REPL should use explicitly provided spark session."""
        from rlm_adk.rlm_repl import RLMREPLEnvironment
        from rlm_adk.runtime import clear_execution_mode_cache

        mock_spark = MagicMock()

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}):
            clear_execution_mode_cache()

            repl = RLMREPLEnvironment(
                llm_query_fn=lambda x: "test",
                context={"data": "test"},
                spark=mock_spark,
            )
            assert repl._spark is mock_spark

    def test_repl_spark_accessible_in_code(self):
        """spark should be accessible in executed code."""
        from rlm_adk.rlm_repl import RLMREPLEnvironment
        from rlm_adk.runtime import clear_execution_mode_cache

        mock_spark = MagicMock()

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}):
            clear_execution_mode_cache()

            repl = RLMREPLEnvironment(
                llm_query_fn=lambda x: "test",
                spark=mock_spark,
            )

            result = repl.execute_code("spark_type = type(spark).__name__")
            assert result["status"] == "success"
            assert repl.get_variable("spark_type") == "MagicMock"

    def test_is_dataframe_context_false_without_spark(self):
        """is_dataframe_context should return False without Spark."""
        from rlm_adk.rlm_repl import RLMREPLEnvironment
        from rlm_adk.runtime import clear_execution_mode_cache

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}):
            clear_execution_mode_cache()

            repl = RLMREPLEnvironment(
                llm_query_fn=lambda x: "test",
                context={"data": "test"},
            )
            assert repl.is_dataframe_context() is False


class TestContextLoaderDualMode:
    """Tests for context loader dual-mode dispatch."""

    def setup_method(self):
        from rlm_adk.runtime import clear_execution_mode_cache

        clear_execution_mode_cache()

    def test_load_vendor_data_local_mode(self):
        """load_vendor_data_to_context should work in local mode."""
        from rlm_adk.runtime import clear_execution_mode_cache
        from rlm_adk.testing import create_tool_context
        from rlm_adk.tools.context_loader import load_vendor_data_to_context

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}, clear=True):
            clear_execution_mode_cache()

            mock_context = create_tool_context({})

            result = load_vendor_data_to_context(
                hospital_chains=["hospital_chain_alpha"],
                include_masterdata=False,
                tool_context=mock_context,
            )
            assert result["status"] == "success"
            assert "chains_loaded" in result

    def test_load_vendor_data_multiple_chains(self):
        """load_vendor_data_to_context should handle multiple chains."""
        from rlm_adk.runtime import clear_execution_mode_cache
        from rlm_adk.testing import create_tool_context
        from rlm_adk.tools.context_loader import load_vendor_data_to_context

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}, clear=True):
            clear_execution_mode_cache()

            mock_context = create_tool_context({})

            result = load_vendor_data_to_context(
                hospital_chains=[
                    "hospital_chain_alpha",
                    "hospital_chain_beta",
                    "hospital_chain_gamma",
                ],
                include_masterdata=False,
                tool_context=mock_context,
            )
            assert result["status"] == "success"
            assert len(result["chains_loaded"]) == 3

    def test_load_vendor_data_with_masterdata(self):
        """load_vendor_data_to_context should include masterdata when requested."""
        from rlm_adk.runtime import clear_execution_mode_cache
        from rlm_adk.testing import create_tool_context
        from rlm_adk.tools.context_loader import load_vendor_data_to_context

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}, clear=True):
            clear_execution_mode_cache()

            mock_context = create_tool_context({})

            result = load_vendor_data_to_context(
                hospital_chains=["hospital_chain_alpha"],
                include_masterdata=True,
                tool_context=mock_context,
            )
            assert result["status"] == "success"
            assert result["masterdata_included"] is True

    def test_load_custom_context(self):
        """load_custom_context should work in local mode."""
        from rlm_adk.runtime import clear_execution_mode_cache
        from rlm_adk.testing import create_tool_context
        from rlm_adk.tools.context_loader import load_custom_context

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}, clear=True):
            clear_execution_mode_cache()

            mock_context = create_tool_context({})

            result = load_custom_context(
                data={"custom": "data", "items": [1, 2, 3]},
                description="Custom test data",
                tool_context=mock_context,
            )
            assert result["status"] == "success"

    def test_load_query_results_to_context(self):
        """load_query_results_to_context should work in local mode."""
        from rlm_adk.runtime import clear_execution_mode_cache
        from rlm_adk.testing import create_tool_context
        from rlm_adk.tools.context_loader import load_query_results_to_context

        with patch.dict(os.environ, {"RLM_EXECUTION_MODE": "local"}, clear=True):
            clear_execution_mode_cache()

            mock_context = create_tool_context({})

            result = load_query_results_to_context(
                sql_query="SELECT * FROM vendors",
                description="All vendors",
                tool_context=mock_context,
            )
            assert result["status"] == "success"
            assert "rows_loaded" in result


class TestNativeModeDetection:
    """Tests for native mode detection when Databricks markers are present."""

    def setup_method(self):
        from rlm_adk.runtime import clear_execution_mode_cache

        clear_execution_mode_cache()

    def test_auto_detect_native_from_runtime_version(self):
        """Should auto-detect native mode from DATABRICKS_RUNTIME_VERSION."""
        from rlm_adk.runtime import clear_execution_mode_cache, get_execution_mode

        with patch.dict(os.environ, {"DATABRICKS_RUNTIME_VERSION": "14.3"}):
            clear_execution_mode_cache()
            assert get_execution_mode() == "native"

    def test_auto_detect_native_from_spark_path(self):
        """Should auto-detect native mode from /databricks/spark path."""
        from rlm_adk.runtime import clear_execution_mode_cache, get_execution_mode

        with patch.dict(os.environ, {}, clear=True):
            clear_execution_mode_cache()
            with patch("os.path.exists", return_value=True):
                clear_execution_mode_cache()
                assert get_execution_mode() == "native"

    def test_explicit_local_overrides_databricks_markers(self):
        """RLM_EXECUTION_MODE=local should override Databricks markers."""
        from rlm_adk.runtime import clear_execution_mode_cache, get_execution_mode

        with patch.dict(
            os.environ,
            {"RLM_EXECUTION_MODE": "local", "DATABRICKS_RUNTIME_VERSION": "14.3"},
        ):
            clear_execution_mode_cache()
            assert get_execution_mode() == "local"


class TestREPLSessionManagement:
    """Tests for REPL session management."""

    def setup_method(self):
        from rlm_adk.rlm_repl import _ACTIVE_REPL_SESSIONS

        _ACTIVE_REPL_SESSIONS.clear()

    def test_get_or_create_repl_session_new(self):
        """get_or_create_repl_session should create new session."""
        from rlm_adk.rlm_repl import get_or_create_repl_session

        session = get_or_create_repl_session(
            session_id="test_session",
            llm_query_fn=lambda x: "test",
            context={"data": "test"},
        )
        assert session is not None
        assert session._context == {"data": "test"}

    def test_get_or_create_repl_session_reuse(self):
        """get_or_create_repl_session should reuse existing session."""
        from rlm_adk.rlm_repl import get_or_create_repl_session

        session1 = get_or_create_repl_session(
            session_id="test_session",
            llm_query_fn=lambda x: "test",
            context={"data": "original"},
        )
        session1.set_variable("x", 42)

        session2 = get_or_create_repl_session(
            session_id="test_session",
            llm_query_fn=lambda x: "test",
            context={"data": "new"},  # Should be ignored
        )

        assert session1 is session2
        assert session2.get_variable("x") == 42

    def test_clear_repl_session(self):
        """clear_repl_session should remove session."""
        from rlm_adk.rlm_repl import (
            _ACTIVE_REPL_SESSIONS,
            clear_repl_session,
            get_or_create_repl_session,
        )

        get_or_create_repl_session(
            session_id="test_session",
            llm_query_fn=lambda x: "test",
        )
        assert "test_session" in _ACTIVE_REPL_SESSIONS

        clear_repl_session("test_session")
        assert "test_session" not in _ACTIVE_REPL_SESSIONS

    def test_clear_nonexistent_session_no_error(self):
        """Clearing non-existent session should not raise error."""
        from rlm_adk.rlm_repl import clear_repl_session

        # Should not raise
        clear_repl_session("nonexistent_session")


class TestCodeBlockParsing:
    """Tests for code block extraction from LM responses."""

    def test_find_code_blocks_python(self):
        """Should extract Python code blocks."""
        from rlm_adk.rlm_repl import find_code_blocks

        response = """
Here's the code:
```python
x = 1 + 1
print(x)
```
        """
        blocks = find_code_blocks(response)
        assert len(blocks) == 1
        assert "x = 1 + 1" in blocks[0]

    def test_find_code_blocks_repl(self):
        """Should extract REPL code blocks."""
        from rlm_adk.rlm_repl import find_code_blocks

        response = """
Execute this:
```repl
result = llm_query("What is 2+2?")
print(result)
```
        """
        blocks = find_code_blocks(response)
        assert len(blocks) == 1
        assert "llm_query" in blocks[0]

    def test_find_code_blocks_multiple(self):
        """Should extract multiple code blocks."""
        from rlm_adk.rlm_repl import find_code_blocks

        response = """
First:
```python
x = 1
```
Then:
```repl
y = llm_query(str(x))
```
        """
        blocks = find_code_blocks(response)
        assert len(blocks) == 2

    def test_find_final_answer(self):
        """Should extract FINAL answer."""
        from rlm_adk.rlm_repl import find_final_answer

        response = "The answer is FINAL(42)"
        answer = find_final_answer(response)
        assert answer == "42"

    def test_find_final_answer_with_quotes(self):
        """Should strip quotes from FINAL answer."""
        from rlm_adk.rlm_repl import find_final_answer

        response = 'The answer is FINAL("hello world")'
        answer = find_final_answer(response)
        assert answer == "hello world"

    def test_find_final_var(self):
        """Should extract FINAL_VAR reference."""
        from rlm_adk.rlm_repl import find_final_answer

        response = "The result is stored in FINAL_VAR(result)"
        answer = find_final_answer(response)
        assert answer == "__FINAL_VAR__:result"

    def test_find_final_answer_none(self):
        """Should return None when no FINAL found."""
        from rlm_adk.rlm_repl import find_final_answer

        response = "No final answer here"
        answer = find_final_answer(response)
        assert answer is None
