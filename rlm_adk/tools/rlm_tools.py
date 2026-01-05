"""RLM-ADK Integration Tools.

These tools bridge the RLM recursive decomposition pattern with Google ADK,
providing the core RLM capabilities as ADK-compatible tools:

1. rlm_execute_code - Execute code with llm_query() and context access
2. rlm_load_context - Load Unity Catalog data into REPL context
3. rlm_query_context - Use RLM pattern to analyze large context
4. rlm_get_session_state - Inspect REPL session state

The LM can use these tools to programmatically decompose large problems
by spawning sub-LM calls from within code execution.
"""

import os
from typing import Any

from google.adk.tools import ToolContext
from rlm_adk.rlm_repl import (
    RLMREPLEnvironment,
    clear_repl_session,
    find_code_blocks,
    find_final_answer,
    get_or_create_repl_session,
)


def _create_llm_query_fn(tool_context: ToolContext):
    """Create an llm_query function that uses the ADK model.

    In a full integration, this would call back to the ADK agent's
    underlying LLM. For now, we use a simulated response for development.
    """
    def llm_query(prompt: str) -> str:
        # Check if we have a real LLM client available
        llm_client = tool_context.state.get("_llm_client")
        if llm_client and hasattr(llm_client, "completion"):
            return llm_client.completion(prompt)

        # Check for Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-3-pro")
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"[LLM Error: {e}]"

        # Simulation mode for development
        return _simulate_llm_response(prompt)

    return llm_query


def _create_llm_query_batched_fn(tool_context: ToolContext):
    """Create a batched llm_query function."""
    llm_query = _create_llm_query_fn(tool_context)

    def llm_query_batched(prompts: list[str]) -> list[str]:
        # Check for async client
        llm_client = tool_context.state.get("_llm_client")
        if llm_client and hasattr(llm_client, "acompletion"):
            import asyncio

            async def run_all():
                tasks = [llm_client.acompletion(p) for p in prompts]
                return await asyncio.gather(*tasks)

            return asyncio.run(run_all())

        # Fallback to sequential (could be parallelized with ThreadPoolExecutor)
        return [llm_query(p) for p in prompts]

    return llm_query_batched


def _simulate_llm_response(prompt: str) -> str:
    """Simulate LLM response for development/testing."""
    prompt_lower = prompt.lower()

    if "vendor" in prompt_lower and "similar" in prompt_lower:
        return "Based on the data, I found 3 similar vendors: MedSupply Corp (Alpha), MedSupply Corporation (Beta), and Med Supply Corp. (Gamma). They appear to be the same company based on matching tax ID 12-3456789."

    if "summarize" in prompt_lower or "summary" in prompt_lower:
        return "Summary: The data contains vendor records from multiple hospital chains. Key attributes include vendor_id, vendor_name, tax_id, and address. There are potential duplicates across chains."

    if "analyze" in prompt_lower:
        return "Analysis: Found 150 total vendors across 3 hospital chains. 45 vendors have matching tax IDs indicating they are the same entity. 23 vendors have similar names but need manual review."

    if "answer" in prompt_lower or "question" in prompt_lower:
        return "Based on the context provided, the answer is: The vendor resolution process identified 45 confirmed matches and 23 potential matches requiring review."

    return f"[Simulated LLM response to: {prompt[:100]}...]"


def rlm_execute_code(
    code: str,
    tool_context: ToolContext,
) -> dict:
    """Execute Python code in the RLM REPL environment.

    This is the core RLM tool - it executes code with access to:
    - `context`: The offloaded context data (e.g., vendor records)
    - `llm_query(prompt)`: Make a recursive sub-LM call
    - `llm_query_batched(prompts)`: Make concurrent sub-LM calls
    - All previously defined variables (persistent across calls)

    Use this to programmatically decompose large problems. For example,
    to analyze millions of vendor records, you can:
    1. Chunk the context
    2. Use llm_query_batched to analyze chunks concurrently
    3. Aggregate results with another llm_query call

    Args:
        code: Python code to execute. Has access to `context`, `llm_query`,
              `llm_query_batched`, and all standard Python libraries.

    Returns:
        dict: Execution results including:
            - 'status': "success" or "error"
            - 'stdout': Captured print output
            - 'stderr': Captured error output
            - 'locals_snapshot': Variables defined/modified
            - 'llm_calls': Number of sub-LM calls made
            - 'error_message': Error details if failed

    Example:
        To analyze vendor data with recursive decomposition:
        ```
        rlm_execute_code('''
        # Context contains list of vendor records
        chunk_size = len(context) // 5
        chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]

        # Analyze chunks concurrently with sub-LM calls
        prompts = [f"Find duplicate vendors in: {chunk}" for chunk in chunks]
        results = llm_query_batched(prompts)

        # Aggregate with final sub-LM call
        final_answer = llm_query(f"Combine these duplicate findings: {results}")
        print(final_answer)
        ''')
        ```
    """
    session_id = tool_context.state.get("rlm_session_id", "default")

    # Get or create the REPL session
    repl = get_or_create_repl_session(
        session_id=session_id,
        llm_query_fn=_create_llm_query_fn(tool_context),
        llm_query_batched_fn=_create_llm_query_batched_fn(tool_context),
        context=tool_context.state.get("rlm_context"),
    )

    # Execute the code
    result = repl.execute_code(code)

    # Store updated context back to state
    tool_context.state["rlm_context"] = repl.context

    return result


def rlm_load_context(
    context_data: Any,
    context_description: str,
    tool_context: ToolContext,
) -> dict:
    """Load data into the RLM REPL context variable.

    This implements the RLM principle of "context offloading" - large
    input data is stored in the REPL's `context` variable where the
    LM can programmatically examine and decompose it.

    Args:
        context_data: The data to load into context. Can be:
            - A list (e.g., list of vendor records)
            - A dict (e.g., structured data from multiple sources)
            - A string (e.g., raw text data)
        context_description: Description of what the context contains.

    Returns:
        dict: Loading results including:
            - 'status': "success" or "error"
            - 'context_type': Type of the loaded context
            - 'context_size': Size/length of the context
            - 'context_preview': Preview of the context data

    Example:
        Load vendor data from multiple hospital chains:
        ```
        rlm_load_context(
            context_data={
                "alpha_vendors": [...],  # 5000 records
                "beta_vendors": [...],   # 3000 records
                "gamma_vendors": [...],  # 4000 records
            },
            context_description="Vendor records from 3 hospital chains for resolution"
        )
        ```
    """
    session_id = tool_context.state.get("rlm_session_id", "default")

    # Store context in tool state
    tool_context.state["rlm_context"] = context_data
    tool_context.state["rlm_context_description"] = context_description

    # Get or create REPL and update its context
    repl = get_or_create_repl_session(
        session_id=session_id,
        llm_query_fn=_create_llm_query_fn(tool_context),
        context=context_data,
    )
    repl.context = context_data

    # Calculate context info
    context_type = type(context_data).__name__
    if isinstance(context_data, (list, dict, str)):
        context_size = len(context_data)
    else:
        context_size = 1

    # Create preview
    try:
        preview = repr(context_data)
        if len(preview) > 500:
            preview = preview[:500] + "..."
    except Exception:
        preview = f"<{context_type} object>"

    return {
        "status": "success",
        "context_type": context_type,
        "context_size": context_size,
        "context_preview": preview,
        "description": context_description,
        "message": f"Loaded {context_type} with {context_size} items into REPL context. "
                   f"Use rlm_execute_code to access via the `context` variable.",
    }


def rlm_query_context(
    query: str,
    strategy: str,
    tool_context: ToolContext,
) -> dict:
    """Use RLM recursive decomposition to analyze the loaded context.

    This is a high-level tool that automatically applies RLM patterns
    to analyze large contexts. It implements the iterative code execution
    loop with sub-LM calls.

    Args:
        query: The question to answer about the context.
        strategy: The decomposition strategy to use:
            - "chunk_and_aggregate": Split context, analyze chunks, aggregate
            - "iterative": Process context iteratively, maintaining state
            - "map_reduce": Map analysis over chunks, reduce to final answer
            - "hierarchical": Build hierarchical summaries

    Returns:
        dict: Analysis results including:
            - 'status': "success" or "error"
            - 'answer': The final answer to the query
            - 'iterations': Number of iterations taken
            - 'llm_calls': Total sub-LM calls made
            - 'reasoning': Step-by-step reasoning trace

    Example:
        Analyze vendor duplicates across hospital chains:
        ```
        rlm_query_context(
            query="Find all duplicate vendors across hospital chains based on tax ID and name similarity",
            strategy="chunk_and_aggregate"
        )
        ```
    """
    session_id = tool_context.state.get("rlm_session_id", "default")
    context = tool_context.state.get("rlm_context")

    if context is None:
        return {
            "status": "error",
            "error_message": "No context loaded. Use rlm_load_context first.",
        }

    repl = get_or_create_repl_session(
        session_id=session_id,
        llm_query_fn=_create_llm_query_fn(tool_context),
        llm_query_batched_fn=_create_llm_query_batched_fn(tool_context),
        context=context,
    )

    reasoning_trace = []
    iterations = 0
    max_iterations = 10

    if strategy == "chunk_and_aggregate":
        code = f'''
# RLM Chunk and Aggregate Strategy
query = """{query}"""

# Determine chunk size based on context
if isinstance(context, list):
    total_items = len(context)
    chunk_size = max(1, total_items // 5)
    chunks = [context[i:i+chunk_size] for i in range(0, total_items, chunk_size)]
elif isinstance(context, dict):
    chunks = [{{k: v}} for k, v in context.items()]
elif isinstance(context, str):
    chunk_size = max(1000, len(context) // 5)
    chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
else:
    chunks = [context]

print(f"Split context into {{len(chunks)}} chunks")

# Analyze chunks concurrently
prompts = [f"Analyze this data to help answer: {{query}}\\n\\nData:\\n{{chunk}}" for chunk in chunks]
chunk_results = llm_query_batched(prompts)

for i, result in enumerate(chunk_results):
    print(f"Chunk {{i+1}} analysis: {{result[:200]}}...")

# Aggregate results
aggregation_prompt = f"""Based on these chunk analyses, provide a final answer to: {{query}}

Chunk Analyses:
{{chr(10).join(f"Chunk {{i+1}}: {{r}}" for i, r in enumerate(chunk_results))}}

Final Answer:"""

final_answer = llm_query(aggregation_prompt)
print(f"\\nFinal Answer: {{final_answer}}")
'''
    elif strategy == "iterative":
        code = f'''
# RLM Iterative Strategy
query = """{query}"""
buffer = []

if isinstance(context, list):
    items = context
elif isinstance(context, dict):
    items = list(context.items())
else:
    items = [context]

for i, item in enumerate(items):
    if i >= 20:  # Limit iterations for safety
        break
    prompt = f"Iteration {{i+1}}/{{len(items)}}. Previous findings: {{buffer[-3:] if buffer else 'None'}}\\n\\nAnalyze this item for: {{query}}\\n\\nItem: {{item}}"
    result = llm_query(prompt)
    buffer.append(result)
    print(f"Iteration {{i+1}}: {{result[:100]}}...")

final_answer = llm_query(f"Summarize findings for {{query}}:\\n{{chr(10).join(buffer)}}")
print(f"\\nFinal Answer: {{final_answer}}")
'''
    elif strategy == "map_reduce":
        code = f'''
# RLM Map-Reduce Strategy
query = """{query}"""

# Map phase
if isinstance(context, list):
    items = context[:50]  # Limit for safety
elif isinstance(context, dict):
    items = list(context.values())[:50]
else:
    items = [context]

map_prompts = [f"Extract relevant info for '{{query}}' from: {{item}}" for item in items]
mapped = llm_query_batched(map_prompts)
print(f"Mapped {{len(mapped)}} items")

# Reduce phase
reduce_prompt = f"Reduce these mapped results to answer '{{query}}':\\n{{chr(10).join(mapped)}}"
final_answer = llm_query(reduce_prompt)
print(f"\\nFinal Answer: {{final_answer}}")
'''
    else:  # hierarchical
        code = f'''
# RLM Hierarchical Strategy
query = """{query}"""

if isinstance(context, dict):
    # Summarize each top-level key
    summaries = {{}}
    for key, value in context.items():
        summary = llm_query(f"Summarize {{key}} data for '{{query}}': {{value}}")
        summaries[key] = summary
        print(f"{{key}} summary: {{summary[:100]}}...")

    # Combine summaries
    final_answer = llm_query(f"Combine summaries to answer '{{query}}':\\n{{summaries}}")
else:
    final_answer = llm_query(f"Answer '{{query}}' based on: {{context}}")

print(f"\\nFinal Answer: {{final_answer}}")
'''

    # Execute the strategy code
    result = repl.execute_code(code)
    iterations += 1

    reasoning_trace.append({
        "iteration": iterations,
        "strategy": strategy,
        "code_executed": code[:500] + "...",
        "result": result,
    })

    # Extract final answer from output
    stdout = result.get("stdout", "")
    final_answer_match = stdout.split("Final Answer:")
    if len(final_answer_match) > 1:
        final_answer = final_answer_match[-1].strip()
    else:
        final_answer = stdout

    stats = repl.get_stats()

    return {
        "status": result.get("status", "success"),
        "answer": final_answer,
        "iterations": iterations,
        "llm_calls": stats["llm_call_count"],
        "reasoning": reasoning_trace,
        "strategy_used": strategy,
    }


def rlm_get_session_state(tool_context: ToolContext) -> dict:
    """Get the current state of the RLM REPL session.

    Use this to inspect what variables are defined, how many sub-LM
    calls have been made, and the current context.

    Returns:
        dict: Session state including:
            - 'status': "success"
            - 'session_id': Current session ID
            - 'variables': Variables defined in REPL
            - 'execution_count': Number of code executions
            - 'llm_call_count': Number of sub-LM calls
            - 'context_loaded': Whether context is loaded
            - 'context_type': Type of loaded context
    """
    session_id = tool_context.state.get("rlm_session_id", "default")
    context = tool_context.state.get("rlm_context")

    repl = get_or_create_repl_session(
        session_id=session_id,
        llm_query_fn=_create_llm_query_fn(tool_context),
        context=context,
    )

    stats = repl.get_stats()

    return {
        "status": "success",
        "session_id": session_id,
        "variables": stats["variables_defined"],
        "execution_count": stats["execution_count"],
        "llm_call_count": stats["llm_call_count"],
        "context_loaded": context is not None,
        "context_type": type(context).__name__ if context else None,
        "context_size": len(context) if context and hasattr(context, "__len__") else None,
    }


def rlm_clear_session(tool_context: ToolContext) -> dict:
    """Clear the RLM REPL session.

    Resets all variables, clears context, and resets counters.

    Returns:
        dict: Status of the clear operation.
    """
    session_id = tool_context.state.get("rlm_session_id", "default")

    clear_repl_session(session_id)
    tool_context.state.pop("rlm_context", None)
    tool_context.state.pop("rlm_context_description", None)

    return {
        "status": "success",
        "message": f"Cleared RLM session '{session_id}'",
    }
