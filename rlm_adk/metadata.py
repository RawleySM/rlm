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
