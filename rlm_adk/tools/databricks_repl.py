"""Databricks REPL tools for Python and SQL execution.

Provides tools for executing Python code and SQL queries in a Databricks
workspace environment with session state management.
"""

import os
from typing import TYPE_CHECKING, Any

from rlm_adk._compat import ToolContextProtocol

if TYPE_CHECKING:
    from rlm_adk._compat import ToolContext

# Session state storage for REPL variables
_REPL_SESSION_STATE: dict[str, Any] = {}


def execute_python_code(code: str, tool_context: ToolContextProtocol) -> dict:
    """Execute Python code in a Databricks workspace REPL environment.

    Use this tool to run Python code for data analysis, transformations,
    and creating visualizations. The code runs in a persistent session
    where variables are preserved between executions.

    Args:
        code: The Python code to execute. Can include pandas, pyspark,
              and other data science libraries available in Databricks.

    Returns:
        dict: Contains execution results.
            - 'status' (str): "success" or "error"
            - 'output' (str, optional): Standard output from execution
            - 'result' (Any, optional): Return value if code returns something
            - 'variables' (list, optional): Names of variables created/modified
            - 'error_message' (str, optional): Error details if execution failed
    """
    try:
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")
        cluster_id = os.getenv("DATABRICKS_CLUSTER_ID")

        if not all([databricks_host, databricks_token, cluster_id]):
            # Simulate execution for development/testing
            return _simulate_python_execution(code, tool_context)

        # In production, use Databricks Command Execution API
        return _execute_databricks_python(code, databricks_host, databricks_token, cluster_id)

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to execute Python code: {str(e)}",
        }


def execute_sql_query(query: str, tool_context: ToolContextProtocol) -> dict:
    """Execute a SQL query against Databricks SQL warehouse or Unity Catalog.

    Use this tool to run SQL queries for data exploration, creating views,
    and performing aggregations across hospital ERP databases and vendor data.

    Args:
        query: The SQL query to execute. Supports Spark SQL syntax and
               Unity Catalog three-level namespace (catalog.schema.table).

    Returns:
        dict: Contains query results.
            - 'status' (str): "success" or "error"
            - 'columns' (list, optional): Column names in result
            - 'data' (list, optional): Query result rows (up to 1000 rows)
            - 'row_count' (int, optional): Total number of rows returned
            - 'execution_time_ms' (float, optional): Query execution time
            - 'error_message' (str, optional): Error details if query failed
    """
    try:
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")
        warehouse_id = os.getenv("DATABRICKS_SQL_WAREHOUSE_ID")

        if not all([databricks_host, databricks_token, warehouse_id]):
            # Simulate execution for development/testing
            return _simulate_sql_execution(query, tool_context)

        # In production, use Databricks SQL Statement Execution API
        return _execute_databricks_sql(query, databricks_host, databricks_token, warehouse_id)

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to execute SQL query: {str(e)}",
        }


def get_repl_session_state(tool_context: ToolContextProtocol) -> dict:
    """Get the current state of variables in the REPL session.

    Use this tool to inspect what variables are currently available
    in the Python REPL session, including their types and values.

    Returns:
        dict: Contains session state information.
            - 'status' (str): "success" or "error"
            - 'variables' (dict, optional): Map of variable names to their info
              Each variable has 'type', 'shape' (if applicable), and 'preview'
            - 'error_message' (str, optional): Error details if failed
    """
    try:
        session_id = tool_context.state.get("repl_session_id", "default")
        state = _REPL_SESSION_STATE.get(session_id, {})

        variables_info = {}
        for name, value in state.items():
            var_info = {"type": type(value).__name__}

            # Add shape for array-like objects
            if hasattr(value, "shape"):
                var_info["shape"] = str(value.shape)
            elif hasattr(value, "__len__") and not isinstance(value, str):
                var_info["length"] = len(value)

            # Add preview
            var_info["preview"] = _get_value_preview(value)
            variables_info[name] = var_info

        return {
            "status": "success",
            "variables": variables_info,
            "session_id": session_id,
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to get session state: {str(e)}",
        }


def _get_value_preview(value: Any, max_length: int = 200) -> str:
    """Get a string preview of a value."""
    try:
        preview = repr(value)
        if len(preview) > max_length:
            preview = preview[: max_length - 3] + "..."
        return preview
    except Exception:
        return f"<{type(value).__name__} object>"


def _simulate_python_execution(code: str, tool_context: ToolContextProtocol) -> dict:
    """Simulate Python execution for development/testing."""
    import io
    import sys

    session_id = tool_context.state.get("repl_session_id", "default")
    if session_id not in _REPL_SESSION_STATE:
        _REPL_SESSION_STATE[session_id] = {}

    local_vars = _REPL_SESSION_STATE[session_id].copy()

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    result = None
    try:
        # Execute the code
        exec_globals = {"__builtins__": __builtins__}
        exec(code, exec_globals, local_vars)

        # Track new/modified variables
        new_vars = [k for k in local_vars if k not in _REPL_SESSION_STATE.get(session_id, {})]
        _REPL_SESSION_STATE[session_id] = local_vars

        output = captured_output.getvalue()

        return {
            "status": "success",
            "output": output if output else None,
            "variables": new_vars if new_vars else None,
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "output": captured_output.getvalue(),
        }
    finally:
        sys.stdout = old_stdout


def _simulate_sql_execution(query: str, tool_context: ToolContextProtocol) -> dict:
    """Simulate SQL execution for development/testing."""
    # Parse the query to provide meaningful mock responses
    query_lower = query.lower().strip()

    if query_lower.startswith("show catalogs"):
        return {
            "status": "success",
            "columns": ["catalog_name"],
            "data": [
                ["healthcare_main"],
                ["hospital_chain_a"],
                ["hospital_chain_b"],
                ["hospital_chain_c"],
                ["masterdata_vendors"],
            ],
            "row_count": 5,
            "execution_time_ms": 45.2,
        }
    elif query_lower.startswith("show schemas"):
        return {
            "status": "success",
            "columns": ["schema_name"],
            "data": [
                ["erp_data"],
                ["vendor_info"],
                ["transactions"],
                ["analytics"],
            ],
            "row_count": 4,
            "execution_time_ms": 32.1,
        }
    elif query_lower.startswith("describe") or query_lower.startswith("desc"):
        return {
            "status": "success",
            "columns": ["col_name", "data_type", "comment"],
            "data": [
                ["vendor_id", "STRING", "Unique vendor identifier"],
                ["vendor_name", "STRING", "Vendor display name"],
                ["tax_id", "STRING", "Federal tax ID"],
                ["address", "STRING", "Primary address"],
                ["created_at", "TIMESTAMP", "Record creation time"],
            ],
            "row_count": 5,
            "execution_time_ms": 28.5,
        }
    elif "select" in query_lower:
        return {
            "status": "success",
            "columns": ["vendor_id", "vendor_name", "hospital_chain"],
            "data": [
                ["V001", "MedSupply Corp", "hospital_chain_a"],
                ["V002", "HealthEquip Inc", "hospital_chain_a"],
                ["V003", "MedSupply Corporation", "hospital_chain_b"],
            ],
            "row_count": 3,
            "execution_time_ms": 156.8,
        }
    elif query_lower.startswith("create"):
        return {
            "status": "success",
            "message": "View/Table created successfully",
            "execution_time_ms": 89.3,
        }
    else:
        return {
            "status": "success",
            "message": "Query executed successfully",
            "execution_time_ms": 50.0,
        }


def _execute_databricks_python(
    code: str, host: str, token: str, cluster_id: str
) -> dict:
    """Execute Python code via Databricks Command Execution API."""
    import requests

    url = f"https://{host}/api/1.2/commands/execute"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "clusterId": cluster_id,
        "language": "python",
        "command": code,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=300)
    response.raise_for_status()
    result = response.json()

    if result.get("status") == "Finished":
        return {
            "status": "success",
            "output": result.get("results", {}).get("data"),
        }
    else:
        return {
            "status": "error",
            "error_message": result.get("results", {}).get("cause", "Unknown error"),
        }


def _execute_databricks_sql(
    query: str, host: str, token: str, warehouse_id: str
) -> dict:
    """Execute SQL via Databricks SQL Statement Execution API."""
    import requests

    url = f"https://{host}/api/2.0/sql/statements"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "warehouse_id": warehouse_id,
        "statement": query,
        "wait_timeout": "30s",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()

    status = result.get("status", {}).get("state")
    if status == "SUCCEEDED":
        manifest = result.get("manifest", {})
        data_chunk = result.get("result", {}).get("data_array", [])

        return {
            "status": "success",
            "columns": [col["name"] for col in manifest.get("schema", {}).get("columns", [])],
            "data": data_chunk,
            "row_count": len(data_chunk),
        }
    else:
        return {
            "status": "error",
            "error_message": result.get("status", {}).get("error", {}).get("message", "Unknown error"),
        }
