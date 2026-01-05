#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=8.0.0",
# ]
# ///
"""Tests for RLM-ADK custom_metadata schemas.

Tests metadata dataclasses used for tracking RLM iteration state,
execution results, sub-LM calls, session metrics, and context descriptions.

Test Classes:
- TestRLMIterationMetadata: Iteration metadata schema (5 tests)
- TestRLMExecutionMetadata: Execution metadata schema (5 tests)
- TestRLMSubLMCallMetadata: Sub-LM call metadata schema (4 tests)
- TestRLMSessionMetrics: Session-level metrics schema (5 tests)
- TestRLMContextMetadata: Context description metadata schema (8 tests)
"""

import pytest

from rlm_adk.metadata import (
    RLMContextMetadata,
    RLMExecutionMetadata,
    RLMIterationMetadata,
    RLMSessionMetrics,
    RLMSubLMCallMetadata,
)


class TestRLMIterationMetadata:
    """Test iteration metadata schema.

    RLMIterationMetadata tracks metadata for a single RLM iteration,
    including code block analysis and detection of special patterns
    (llm_query, FINAL, FINAL_VAR).
    """

    def test_rlm_iteration_metadata_defaults(self):
        """Default values are correct."""
        metadata = RLMIterationMetadata(iteration_number=1)

        assert metadata.iteration_number == 1
        assert metadata.code_block_count == 0
        assert metadata.has_llm_query is False
        assert metadata.has_llm_query_batched is False
        assert metadata.has_final_answer is False
        assert metadata.final_answer_type is None

    def test_rlm_iteration_metadata_all_fields(self):
        """All fields assignable."""
        metadata = RLMIterationMetadata(
            iteration_number=3,
            code_block_count=2,
            has_llm_query=True,
            has_llm_query_batched=True,
            has_final_answer=True,
            final_answer_type="FINAL",
        )

        assert metadata.iteration_number == 3
        assert metadata.code_block_count == 2
        assert metadata.has_llm_query is True
        assert metadata.has_llm_query_batched is True
        assert metadata.has_final_answer is True
        assert metadata.final_answer_type == "FINAL"

    def test_rlm_iteration_metadata_to_dict(self):
        """Serializes to dict correctly."""
        metadata = RLMIterationMetadata(
            iteration_number=2,
            code_block_count=1,
            has_llm_query=True,
        )

        result = metadata.to_dict()

        assert isinstance(result, dict)
        assert "iteration_number" in result
        assert "code_block_count" in result
        assert "has_llm_query" in result
        assert "has_llm_query_batched" in result
        assert "has_final_answer" in result
        assert "final_answer_type" in result
        assert result["iteration_number"] == 2
        assert result["code_block_count"] == 1
        assert result["has_llm_query"] is True

    def test_rlm_iteration_metadata_to_dict_includes_iteration_number(self):
        """Dict has iteration_number."""
        metadata = RLMIterationMetadata(iteration_number=5)

        result = metadata.to_dict()

        assert "iteration_number" in result
        assert result["iteration_number"] == 5

    def test_rlm_iteration_metadata_final_answer_types(self):
        """final_answer_type accepts valid values."""
        # Test with FINAL
        metadata_final = RLMIterationMetadata(
            iteration_number=1,
            final_answer_type="FINAL"
        )
        assert metadata_final.final_answer_type == "FINAL"

        # Test with FINAL_VAR
        metadata_final_var = RLMIterationMetadata(
            iteration_number=1,
            final_answer_type="FINAL_VAR"
        )
        assert metadata_final_var.final_answer_type == "FINAL_VAR"

        # Test with None (default)
        metadata_none = RLMIterationMetadata(iteration_number=1)
        assert metadata_none.final_answer_type is None


class TestRLMExecutionMetadata:
    """Test execution metadata schema.

    RLMExecutionMetadata tracks code execution results including
    blocks executed, LLM calls made, timing, status, and errors.
    """

    def test_rlm_execution_metadata_defaults(self):
        """Default values are correct."""
        metadata = RLMExecutionMetadata(iteration_number=1)

        assert metadata.iteration_number == 1
        assert metadata.blocks_executed == 0
        assert metadata.llm_calls_made == 0
        assert metadata.execution_time_ms == 0.0
        assert metadata.status == "success"
        assert metadata.error_type is None
        assert metadata.variables_created == []

    def test_rlm_execution_metadata_all_fields(self):
        """All fields assignable."""
        metadata = RLMExecutionMetadata(
            iteration_number=2,
            blocks_executed=3,
            llm_calls_made=5,
            execution_time_ms=123.45,
            status="error",
            error_type="NameError",
            variables_created=["result", "data", "analysis"],
        )

        assert metadata.iteration_number == 2
        assert metadata.blocks_executed == 3
        assert metadata.llm_calls_made == 5
        assert metadata.execution_time_ms == 123.45
        assert metadata.status == "error"
        assert metadata.error_type == "NameError"
        assert metadata.variables_created == ["result", "data", "analysis"]

    def test_rlm_execution_metadata_to_dict(self):
        """Serializes to dict correctly."""
        metadata = RLMExecutionMetadata(
            iteration_number=1,
            blocks_executed=2,
            llm_calls_made=1,
        )

        result = metadata.to_dict()

        assert isinstance(result, dict)
        assert "iteration_number" in result
        assert "blocks_executed" in result
        assert "llm_calls_made" in result
        assert "execution_time_ms" in result
        assert "status" in result
        assert "error_type" in result
        assert "variables_created" in result
        assert result["iteration_number"] == 1
        assert result["blocks_executed"] == 2

    def test_rlm_execution_metadata_variables_created_mutable(self):
        """variables_created list is mutable."""
        metadata = RLMExecutionMetadata(iteration_number=1)

        # Verify starts empty
        assert metadata.variables_created == []

        # Append items
        metadata.variables_created.append("var1")
        metadata.variables_created.append("var2")

        assert len(metadata.variables_created) == 2
        assert "var1" in metadata.variables_created
        assert "var2" in metadata.variables_created

    def test_rlm_execution_metadata_status_values(self):
        """status accepts valid values."""
        # Test success status
        metadata_success = RLMExecutionMetadata(
            iteration_number=1,
            status="success"
        )
        assert metadata_success.status == "success"

        # Test error status
        metadata_error = RLMExecutionMetadata(
            iteration_number=1,
            status="error"
        )
        assert metadata_error.status == "error"

        # Test no_code status
        metadata_no_code = RLMExecutionMetadata(
            iteration_number=1,
            status="no_code"
        )
        assert metadata_no_code.status == "no_code"


class TestRLMSubLMCallMetadata:
    """Test sub-LM call metadata schema.

    RLMSubLMCallMetadata tracks individual llm_query() or llm_query_batched()
    calls made during code execution, including latency and batch information.
    """

    def test_rlm_sub_lm_call_metadata_defaults(self):
        """Default values for batching."""
        metadata = RLMSubLMCallMetadata(
            call_index=0,
            prompt_length=100,
            response_length=200,
            latency_ms=150.5,
        )

        assert metadata.call_index == 0
        assert metadata.prompt_length == 100
        assert metadata.response_length == 200
        assert metadata.latency_ms == 150.5
        assert metadata.is_batched is False
        assert metadata.batch_size == 1

    def test_rlm_sub_lm_call_metadata_all_fields(self):
        """All fields assignable."""
        metadata = RLMSubLMCallMetadata(
            call_index=2,
            prompt_length=500,
            response_length=1000,
            latency_ms=250.75,
            is_batched=True,
            batch_size=10,
        )

        assert metadata.call_index == 2
        assert metadata.prompt_length == 500
        assert metadata.response_length == 1000
        assert metadata.latency_ms == 250.75
        assert metadata.is_batched is True
        assert metadata.batch_size == 10

    def test_rlm_sub_lm_call_metadata_to_dict(self):
        """Serializes correctly."""
        metadata = RLMSubLMCallMetadata(
            call_index=1,
            prompt_length=300,
            response_length=600,
            latency_ms=175.25,
        )

        result = metadata.to_dict()

        assert isinstance(result, dict)
        assert "call_index" in result
        assert "prompt_length" in result
        assert "response_length" in result
        assert "latency_ms" in result
        assert "is_batched" in result
        assert "batch_size" in result
        assert result["call_index"] == 1
        assert result["prompt_length"] == 300
        assert result["response_length"] == 600
        assert result["latency_ms"] == 175.25

    def test_rlm_sub_lm_call_metadata_batch_tracking(self):
        """Tracks batch information."""
        # Test non-batched call
        single_call = RLMSubLMCallMetadata(
            call_index=0,
            prompt_length=100,
            response_length=200,
            latency_ms=100.0,
            is_batched=False,
            batch_size=1,
        )
        assert single_call.is_batched is False
        assert single_call.batch_size == 1

        # Test batched call
        batch_call = RLMSubLMCallMetadata(
            call_index=1,
            prompt_length=500,
            response_length=1000,
            latency_ms=300.0,
            is_batched=True,
            batch_size=5,
        )
        assert batch_call.is_batched is True
        assert batch_call.batch_size == 5


class TestRLMSessionMetrics:
    """Test session-level metrics schema.

    RLMSessionMetrics aggregates metrics across all iterations in an RLM session,
    tracking totals, timing, errors, and termination information.
    """

    def test_rlm_session_metrics_requires_session_id(self):
        """session_id is required."""
        # Creating without session_id should raise TypeError
        with pytest.raises(TypeError, match="session_id"):
            RLMSessionMetrics()  # type: ignore

    def test_rlm_session_metrics_defaults(self):
        """All counters default to 0."""
        metrics = RLMSessionMetrics(session_id="test_session")

        assert metrics.session_id == "test_session"
        assert metrics.total_iterations == 0
        assert metrics.total_code_blocks == 0
        assert metrics.total_llm_query_calls == 0
        assert metrics.total_llm_query_batched_calls == 0
        assert metrics.total_sub_lm_calls == 0
        assert metrics.total_execution_time_seconds == 0.0
        assert metrics.total_model_latency_ms == 0.0
        assert metrics.errors_encountered == 0
        assert metrics.errors_recovered == 0
        assert metrics.final_answer_found is False
        assert metrics.termination_reason == ""

    def test_rlm_session_metrics_to_dict(self):
        """Serializes correctly."""
        metrics = RLMSessionMetrics(
            session_id="test_session",
            total_iterations=5,
            total_code_blocks=10,
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert "session_id" in result
        assert "total_iterations" in result
        assert "total_code_blocks" in result
        assert "total_llm_query_calls" in result
        assert "total_llm_query_batched_calls" in result
        assert "total_sub_lm_calls" in result
        assert "total_execution_time_seconds" in result
        assert "total_model_latency_ms" in result
        assert "errors_encountered" in result
        assert "errors_recovered" in result
        assert "final_answer_found" in result
        assert "termination_reason" in result
        assert result["session_id"] == "test_session"
        assert result["total_iterations"] == 5

    def test_rlm_session_metrics_termination_reasons(self):
        """termination_reason accepts values."""
        # Test FINAL termination
        metrics_final = RLMSessionMetrics(
            session_id="session1",
            termination_reason="FINAL"
        )
        assert metrics_final.termination_reason == "FINAL"

        # Test max_iterations termination
        metrics_max = RLMSessionMetrics(
            session_id="session2",
            termination_reason="max_iterations"
        )
        assert metrics_max.termination_reason == "max_iterations"

        # Test error termination
        metrics_error = RLMSessionMetrics(
            session_id="session3",
            termination_reason="error"
        )
        assert metrics_error.termination_reason == "error"

    def test_rlm_session_metrics_aggregation(self):
        """Fields can be incremented."""
        metrics = RLMSessionMetrics(session_id="test_session")

        # Verify initial state
        assert metrics.total_iterations == 0
        assert metrics.total_code_blocks == 0
        assert metrics.total_sub_lm_calls == 0

        # Increment values
        metrics.total_iterations += 1
        metrics.total_code_blocks += 3
        metrics.total_sub_lm_calls += 2
        metrics.total_execution_time_seconds += 1.5

        # Verify updates
        assert metrics.total_iterations == 1
        assert metrics.total_code_blocks == 3
        assert metrics.total_sub_lm_calls == 2
        assert metrics.total_execution_time_seconds == 1.5


class TestRLMContextMetadata:
    """Test context description metadata schema.

    RLMContextMetadata describes the loaded context data, including type,
    size, chunking information, data sources, and optional schema hints.
    """

    def test_rlm_context_metadata_required_fields(self):
        """context_type and total_size_chars required."""
        # Creating without required fields should raise TypeError
        with pytest.raises(TypeError):
            RLMContextMetadata()  # type: ignore

        # Creating with only context_type should fail
        with pytest.raises(TypeError):
            RLMContextMetadata(context_type="dict")  # type: ignore

        # Creating with both required fields should succeed
        metadata = RLMContextMetadata(
            context_type="dict",
            total_size_chars=1000
        )
        assert metadata.context_type == "dict"
        assert metadata.total_size_chars == 1000

    def test_rlm_context_metadata_defaults(self):
        """List fields default to empty."""
        metadata = RLMContextMetadata(
            context_type="list",
            total_size_chars=5000
        )

        assert metadata.context_type == "list"
        assert metadata.total_size_chars == 5000
        assert metadata.chunk_count == 0
        assert metadata.chunk_sizes == []
        assert metadata.data_sources == []
        assert metadata.schema_hint is None

    def test_rlm_context_metadata_to_dict(self):
        """Serializes correctly."""
        metadata = RLMContextMetadata(
            context_type="dict",
            total_size_chars=10000,
            chunk_count=3,
            chunk_sizes=[3000, 3500, 3500],
            data_sources=["hospital_chain_alpha", "masterdata"],
        )

        result = metadata.to_dict()

        assert isinstance(result, dict)
        assert "context_type" in result
        assert "total_size_chars" in result
        assert "chunk_count" in result
        assert "chunk_sizes" in result
        assert "data_sources" in result
        assert "schema_hint" in result
        assert result["context_type"] == "dict"
        assert result["total_size_chars"] == 10000
        assert result["chunk_count"] == 3

    def test_rlm_context_metadata_format_for_prompt_basic(self):
        """Basic formatting works."""
        metadata = RLMContextMetadata(
            context_type="dict",
            total_size_chars=1000
        )

        result = metadata.format_for_prompt()

        assert isinstance(result, str)
        assert "dict" in result
        assert "1,000" in result  # Check comma formatting
        assert "Context type:" in result
        assert "Total size:" in result

    def test_rlm_context_metadata_format_for_prompt_with_chunks(self):
        """Includes chunk info."""
        metadata = RLMContextMetadata(
            context_type="list",
            total_size_chars=5000,
            chunk_count=2,
            chunk_sizes=[2500, 2500],
        )

        result = metadata.format_for_prompt()

        assert "Chunks: 2" in result
        assert "Chunk sizes:" in result
        assert "2500" in result

    def test_rlm_context_metadata_format_for_prompt_with_sources(self):
        """Includes data sources."""
        metadata = RLMContextMetadata(
            context_type="dict",
            total_size_chars=3000,
            data_sources=["hospital_chain_alpha", "hospital_chain_beta", "masterdata"],
        )

        result = metadata.format_for_prompt()

        assert "Data sources:" in result
        assert "hospital_chain_alpha" in result
        assert "hospital_chain_beta" in result
        assert "masterdata" in result

    def test_rlm_context_metadata_format_for_prompt_truncates_many_chunks(self):
        """Truncates long chunk lists."""
        # Create metadata with more than 10 chunks
        chunk_sizes = [1000] * 15
        metadata = RLMContextMetadata(
            context_type="list",
            total_size_chars=15000,
            chunk_count=15,
            chunk_sizes=chunk_sizes,
        )

        result = metadata.format_for_prompt()

        assert "Chunks: 15" in result
        assert "more" in result  # Should show truncation indicator
        # Should show first 5 chunks with truncation indicator
        assert "1000, 1000, 1000, 1000, 1000" in result
        # Should not show all 15 values
        assert result.count("1000") == 5  # Only first 5 shown

    def test_rlm_context_metadata_schema_hint_optional(self):
        """schema_hint can be None or dict."""
        # Test with None (default)
        metadata_none = RLMContextMetadata(
            context_type="dict",
            total_size_chars=1000
        )
        assert metadata_none.schema_hint is None

        # Test with dict
        schema = {"vendors": "list[dict]", "masterdata": "dict"}
        metadata_dict = RLMContextMetadata(
            context_type="dict",
            total_size_chars=1000,
            schema_hint=schema
        )
        assert metadata_dict.schema_hint == schema
        assert "vendors" in metadata_dict.schema_hint
        assert metadata_dict.schema_hint["vendors"] == "list[dict]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
