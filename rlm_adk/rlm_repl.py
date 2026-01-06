"""RLM REPL Environment for ADK Integration.

This module provides an RLM-style REPL environment that can be used within
Google ADK agents. It honors the fundamental RLM principles:

1. Context variable - Large input data offloaded to REPL's `context` variable
2. llm_query() - Recursive sub-LM calls from within code execution
3. llm_query_batched() - Concurrent sub-LM calls for parallel decomposition
4. Iterative execution - Code blocks executed iteratively until FINAL answer

This enables LMs to programmatically decompose large problems (like vendor
resolution across millions of hospital records) by spawning sub-LM calls.
"""

import io
import re
import sys
from collections.abc import Callable
from typing import Any

from rlm_adk.runtime import get_execution_mode, get_spark_session


class RLMREPLEnvironment:
    """RLM-style REPL environment with llm_query support.

    This environment provides the core RLM functionality:
    - `context` variable for offloaded data
    - `llm_query(prompt)` for recursive sub-LM calls
    - `llm_query_batched(prompts)` for concurrent sub-LM calls
    - Persistent state across code executions
    """

    def __init__(
        self,
        llm_query_fn: Callable[[str], str],
        llm_query_batched_fn: Callable[[list[str]], list[str]] | None = None,
        context: Any = None,
        spark: Any = None,
    ):
        """Initialize the RLM REPL environment.

        Args:
            llm_query_fn: Function to call for sub-LM queries.
                          Signature: (prompt: str) -> str
            llm_query_batched_fn: Optional function for batched queries.
                                  Signature: (prompts: list[str]) -> list[str]
                                  If not provided, falls back to sequential llm_query calls.
            context: Initial context data to offload to the REPL.
            spark: Optional SparkSession for native execution mode.
        """
        self._llm_query_fn = llm_query_fn
        self._llm_query_batched_fn = llm_query_batched_fn
        self._context = context
        self._local_vars: dict[str, Any] = {}
        self._execution_count = 0
        self._llm_call_count = 0

        # Spark injection: use provided session, auto-get in native mode, or None
        if spark is not None:
            self._spark = spark
        elif get_execution_mode() == "native":
            self._spark = get_spark_session()
        else:
            self._spark = None

    @property
    def context(self) -> Any:
        """Get the current context with lazy DataFrame resolution."""
        # Lazy resolution: if context is a table name string, load it
        if isinstance(self._context, str) and self._spark is not None:
            # Check if it looks like a table reference (catalog.schema.table)
            if "." in self._context:
                try:
                    return self._spark.table(self._context)
                except Exception:
                    pass  # Fall through to return as-is
        return self._context

    @context.setter
    def context(self, value: Any) -> None:
        """Set the context."""
        self._context = value

    def is_dataframe_context(self) -> bool:
        """Check if context is a Spark DataFrame."""
        if self._spark is None:
            return False
        try:
            from pyspark.sql import DataFrame

            return isinstance(self.context, DataFrame)
        except ImportError:
            return False

    def llm_query(self, prompt: str, model: str | None = None) -> str:
        """Make a recursive sub-LM call.

        This is the core RLM mechanism - the LM can spawn sub-LM calls
        from within code execution to process large contexts.

        Args:
            prompt: The prompt to send to the sub-LM.
            model: Optional model name override.

        Returns:
            The sub-LM's response as a string.
        """
        self._llm_call_count += 1
        return self._llm_query_fn(prompt)

    def llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        """Make concurrent sub-LM calls.

        Much faster than sequential llm_query calls when you have
        multiple independent queries. Results are returned in the
        same order as the input prompts.

        Args:
            prompts: List of prompts to send concurrently.
            model: Optional model name override.

        Returns:
            List of responses in the same order as input prompts.
        """
        if self._llm_query_batched_fn:
            self._llm_call_count += len(prompts)
            return self._llm_query_batched_fn(prompts)
        else:
            # Fallback to sequential calls
            return [self.llm_query(p, model) for p in prompts]

    def execute_code(self, code: str) -> dict:
        """Execute Python code in the REPL environment.

        The code has access to:
        - `context`: The offloaded context data
        - `llm_query(prompt)`: Make a sub-LM call
        - `llm_query_batched(prompts)`: Make concurrent sub-LM calls
        - All previously defined variables

        Args:
            code: Python code to execute.

        Returns:
            dict with 'status', 'stdout', 'stderr', 'locals_snapshot', 'llm_calls'
        """
        self._execution_count += 1
        llm_calls_before = self._llm_call_count

        # Set up the execution environment with RLM functions
        # Use a single namespace to avoid Python's exec() scoping issues with
        # comprehensions (they can only see globals, not locals in separate dicts)
        exec_namespace = {
            "__builtins__": __builtins__,
            "context": self._context,  # Use self._context to avoid lazy eval here
            "llm_query": self.llm_query,
            "llm_query_batched": self.llm_query_batched,
            "spark": self._spark,  # Inject SparkSession
        }

        # Include previously defined variables in the same namespace
        exec_namespace.update(self._local_vars)

        # Capture stdout/stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = captured_stdout = io.StringIO()
        sys.stderr = captured_stderr = io.StringIO()

        try:
            exec(code, exec_namespace)

            # Update persistent local variables (excluding builtins and RLM functions)
            reserved_keys = {"__builtins__", "context", "llm_query", "llm_query_batched", "spark"}
            for key, value in exec_namespace.items():
                if not key.startswith("_") and key not in reserved_keys:
                    self._local_vars[key] = value

            stdout = captured_stdout.getvalue()
            stderr = captured_stderr.getvalue()

            # Create snapshot of interesting locals (exclude large objects)
            locals_snapshot = {}
            for k, v in self._local_vars.items():
                try:
                    preview = repr(v)
                    if len(preview) > 500:
                        preview = preview[:500] + "..."
                    locals_snapshot[k] = {
                        "type": type(v).__name__,
                        "preview": preview,
                    }
                except Exception:
                    locals_snapshot[k] = {"type": type(v).__name__, "preview": "<unprintable>"}

            return {
                "status": "success",
                "stdout": stdout,
                "stderr": stderr,
                "locals_snapshot": locals_snapshot,
                "llm_calls": self._llm_call_count - llm_calls_before,
            }

        except Exception as e:
            return {
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
                "llm_calls": self._llm_call_count - llm_calls_before,
            }
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def get_variable(self, name: str) -> Any:
        """Get a variable from the REPL environment."""
        return self._local_vars.get(name)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the REPL environment."""
        self._local_vars[name] = value

    def get_stats(self) -> dict:
        """Get execution statistics."""
        return {
            "execution_count": self._execution_count,
            "llm_call_count": self._llm_call_count,
            "variables_defined": list(self._local_vars.keys()),
        }

    def reset(self) -> None:
        """Reset the REPL environment."""
        self._local_vars = {}
        self._execution_count = 0
        self._llm_call_count = 0


def find_code_blocks(response: str) -> list[str]:
    """Extract code blocks from LM response.

    Looks for ```repl or ```python code blocks.

    Args:
        response: The LM's response text.

    Returns:
        List of code strings extracted from code blocks.
    """
    # Match ```repl or ```python code blocks
    pattern = r"```(?:repl|python)\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    return matches


def find_final_answer(response: str) -> str | None:
    """Extract final answer from LM response.

    Looks for FINAL(answer) or FINAL_VAR(variable_name) patterns.

    Args:
        response: The LM's response text.

    Returns:
        The final answer string, or None if not found.
    """
    # Check for FINAL(...)
    final_match = re.search(r"FINAL\((.*?)\)", response, re.DOTALL)
    if final_match:
        content = final_match.group(1).strip()
        # Strip quotes if present
        if (content.startswith('"') and content.endswith('"')) or \
           (content.startswith("'") and content.endswith("'")):
            content = content[1:-1]
        return content.strip()

    # Check for FINAL_VAR(variable_name) - this needs to be resolved later
    final_var_match = re.search(r"FINAL_VAR\((\w+)\)", response)
    if final_var_match:
        return f"__FINAL_VAR__:{final_var_match.group(1)}"

    return None


# Global REPL instance for session persistence across tool calls
_ACTIVE_REPL_SESSIONS: dict[str, RLMREPLEnvironment] = {}


def get_or_create_repl_session(
    session_id: str,
    llm_query_fn: Callable[[str], str],
    llm_query_batched_fn: Callable[[list[str]], list[str]] | None = None,
    context: Any = None,
    spark: Any = None,
) -> RLMREPLEnvironment:
    """Get or create an RLM REPL session.

    Sessions are persistent across tool calls within the same conversation.

    Args:
        session_id: Unique identifier for the session.
        llm_query_fn: Function for sub-LM calls.
        llm_query_batched_fn: Optional function for batched calls.
        context: Initial context (only used when creating new session).
        spark: Optional SparkSession (only used when creating new session).

    Returns:
        The RLM REPL environment for this session.
    """
    if session_id not in _ACTIVE_REPL_SESSIONS:
        _ACTIVE_REPL_SESSIONS[session_id] = RLMREPLEnvironment(
            llm_query_fn=llm_query_fn,
            llm_query_batched_fn=llm_query_batched_fn,
            context=context,
            spark=spark,
        )
    return _ACTIVE_REPL_SESSIONS[session_id]


def clear_repl_session(session_id: str) -> None:
    """Clear an RLM REPL session."""
    if session_id in _ACTIVE_REPL_SESSIONS:
        del _ACTIVE_REPL_SESSIONS[session_id]
