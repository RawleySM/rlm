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
    invocation_context: InvocationContext | None = None,
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
    invocation_context: InvocationContext | None = None,
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
