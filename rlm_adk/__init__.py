"""RLM-ADK: Recursive Language Model Agent using Google ADK.

A data scientist agent that uses a Databricks workspace Python REPL environment
to generate views and perform tasks across Unity Catalog volumes featuring
hospital chain ERP databases and masterdata vendor resolution.

Usage:
    # Import the root agent
    from rlm_adk import root_agent

    # Tools require google-adk installed
    from rlm_adk.tools import list_catalogs, execute_sql_query

Installation:
    pip install rlm[rlm-adk]  # Installs google-adk and dependencies
"""

from rlm_adk.agent import (
    parallel_erp_analysis,
    root_agent,
    vendor_resolution_pipeline,
)

__all__ = [
    "root_agent",
    "vendor_resolution_pipeline",
    "parallel_erp_analysis",
]
