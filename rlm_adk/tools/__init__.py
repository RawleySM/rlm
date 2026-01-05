"""RLM-ADK Tools.

Custom tools for:
- RLM REPL with llm_query() for recursive decomposition
- Databricks REPL for Python/SQL execution
- Unity Catalog for data exploration
- Vendor resolution for masterdata management
- Context loading for offloading large datasets
"""

from rlm_adk.tools.databricks_repl import (
    execute_python_code,
    execute_sql_query,
    get_repl_session_state,
)
from rlm_adk.tools.unity_catalog import (
    create_view,
    get_volume_metadata,
    list_catalogs,
    list_schemas,
    list_tables,
    list_volumes,
    read_table_sample,
)
from rlm_adk.tools.vendor_resolution import (
    create_vendor_mapping,
    find_similar_vendors,
    get_masterdata_vendor,
    resolve_vendor_to_masterdata,
    search_vendor_by_attributes,
)
from rlm_adk.tools.rlm_tools import (
    rlm_execute_code,
    rlm_load_context,
    rlm_query_context,
    rlm_get_session_state,
    rlm_clear_session,
)
from rlm_adk.tools.context_loader import (
    load_vendor_data_to_context,
    load_custom_context,
    load_query_results_to_context,
)

__all__ = [
    # RLM REPL tools (core recursive decomposition)
    "rlm_execute_code",
    "rlm_load_context",
    "rlm_query_context",
    "rlm_get_session_state",
    "rlm_clear_session",
    # Context loading tools
    "load_vendor_data_to_context",
    "load_custom_context",
    "load_query_results_to_context",
    # Databricks REPL tools
    "execute_python_code",
    "execute_sql_query",
    "get_repl_session_state",
    # Unity Catalog tools
    "list_catalogs",
    "list_schemas",
    "list_tables",
    "list_volumes",
    "get_volume_metadata",
    "read_table_sample",
    "create_view",
    # Vendor resolution tools
    "find_similar_vendors",
    "resolve_vendor_to_masterdata",
    "get_masterdata_vendor",
    "create_vendor_mapping",
    "search_vendor_by_attributes",
]
