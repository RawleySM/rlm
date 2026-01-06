"""Context Loader for Unity Catalog to RLM REPL Integration.

This module provides tools to load data from Unity Catalog (hospital ERP
databases, vendor masterdata) into the RLM REPL context for recursive
decomposition analysis.

The context loading implements the RLM principle of "offloading" - large
datasets are stored in the REPL's `context` variable where the LM can
programmatically examine and decompose them using llm_query() calls.
"""

import os
from typing import Any

from google.adk.tools import ToolContext

from rlm_adk.runtime import get_execution_mode, get_spark_session
from rlm_adk.tools.rlm_tools import rlm_load_context


def _load_vendor_data_native(
    hospital_chains: list[str],
    include_masterdata: bool,
    tool_context: ToolContext,
) -> dict:
    """Load vendor data as Spark DataFrames in native mode."""
    spark = get_spark_session()

    # Create a UNION of all vendor tables
    union_parts = []
    for chain in hospital_chains:
        # Add source chain identifier
        union_parts.append(
            f"SELECT *, '{chain}' as source_chain FROM {chain}.erp_vendors.vendors"
        )

    if union_parts:
        union_query = " UNION ALL ".join(union_parts)
        vendor_df = spark.sql(union_query)

        # Create temp view for easy access
        vendor_df.createOrReplaceTempView("rlm_vendor_context")

        # Get count without collecting all data
        vendor_count = vendor_df.count()
    else:
        vendor_df = None
        vendor_count = 0

    # Load masterdata if requested
    masterdata_df = None
    if include_masterdata:
        try:
            masterdata_df = spark.table("masterdata_vendors.golden_records.vendors")
            masterdata_df.createOrReplaceTempView("rlm_masterdata_context")
        except Exception:
            pass  # Masterdata table may not exist

    # Build context as dict of DataFrames (not collected data!)
    context_data = {"vendors": vendor_df}
    if masterdata_df is not None:
        context_data["masterdata"] = masterdata_df

    # Load into RLM context
    description = (
        f"Spark DataFrame with {vendor_count} vendors from {len(hospital_chains)} chains"
        + (" with masterdata" if include_masterdata and masterdata_df else "")
    )

    load_result = rlm_load_context(
        context_data=context_data,
        context_description=description,
        tool_context=tool_context,
    )

    return {
        "status": "success",
        "context_summary": description,
        "total_vendors": vendor_count,
        "chains_loaded": hospital_chains,
        "masterdata_included": include_masterdata and masterdata_df is not None,
        "context_type": "spark_dataframe",
        "temp_views_created": ["rlm_vendor_context"] + (["rlm_masterdata_context"] if masterdata_df else []),
        "rlm_load_result": load_result,
        "next_step": "Use rlm_execute_code with spark.sql() or DataFrame operations",
    }


def load_vendor_data_to_context(
    hospital_chains: list[str],
    include_masterdata: bool,
    tool_context: ToolContext,
) -> dict:
    """Load vendor data from multiple hospital chains into RLM context.

    This is the primary context loading tool for vendor resolution tasks.
    It queries Unity Catalog tables from each hospital chain and loads
    the combined data into the RLM REPL context.

    The loaded context will be a dict with structure:
    {
        "hospital_chain_alpha": {"vendors": [...], "addresses": [...]},
        "hospital_chain_beta": {"vendors": [...], "addresses": [...]},
        "masterdata": {"golden_records": [...], "mappings": [...]}  # if included
    }

    Args:
        hospital_chains: List of hospital chain catalog names to load.
            Example: ["hospital_chain_alpha", "hospital_chain_beta"]
        include_masterdata: Whether to include masterdata vendor records.

    Returns:
        dict: Loading results including:
            - 'status': "success" or "error"
            - 'context_summary': Summary of loaded data
            - 'total_vendors': Total vendor count across all chains
            - 'chains_loaded': List of chains successfully loaded

    Example:
        Load data for vendor resolution:
        ```
        load_vendor_data_to_context(
            hospital_chains=["hospital_chain_alpha", "hospital_chain_beta", "hospital_chain_gamma"],
            include_masterdata=True
        )
        ```
        Then use rlm_execute_code to analyze with llm_query().
    """
    # Check for native mode first
    if get_execution_mode() == "native":
        return _load_vendor_data_native(hospital_chains, include_masterdata, tool_context)

    # Existing local mode implementation

    context_data = {}
    total_vendors = 0
    chains_loaded = []

    try:
        # Load data from each hospital chain
        for chain in hospital_chains:
            chain_data = _load_chain_data(chain, tool_context)
            if chain_data:
                context_data[chain] = chain_data
                total_vendors += len(chain_data.get("vendors", []))
                chains_loaded.append(chain)

        # Load masterdata if requested
        if include_masterdata:
            masterdata = _load_masterdata(tool_context)
            if masterdata:
                context_data["masterdata"] = masterdata

        # Load into RLM context
        description = (
            f"Vendor data from {len(chains_loaded)} hospital chains "
            f"({total_vendors} total vendors)"
            + (" with masterdata golden records" if include_masterdata else "")
        )

        load_result = rlm_load_context(
            context_data=context_data,
            context_description=description,
            tool_context=tool_context,
        )

        return {
            "status": "success",
            "context_summary": description,
            "total_vendors": total_vendors,
            "chains_loaded": chains_loaded,
            "masterdata_included": include_masterdata,
            "context_structure": {
                chain: list(data.keys()) for chain, data in context_data.items()
            },
            "rlm_load_result": load_result,
            "next_step": "Use rlm_execute_code or rlm_query_context to analyze the loaded data with llm_query() calls",
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to load vendor data: {str(e)}",
        }


def _load_chain_data(chain_name: str, tool_context: ToolContext) -> dict | None:
    """Load vendor data from a hospital chain catalog."""
    # Check for Databricks connection
    databricks_host = os.getenv("DATABRICKS_HOST")

    if not databricks_host:
        # Simulation mode
        return _simulate_chain_data(chain_name)

    # Real Databricks mode - query actual tables
    from rlm_adk.tools.databricks_repl import execute_sql_query

    chain_data = {}

    # Query vendors table
    vendors_query = f"SELECT * FROM {chain_name}.erp_vendors.vendors LIMIT 1000"
    result = execute_sql_query(vendors_query, tool_context)

    if result.get("status") == "success":
        chain_data["vendors"] = result.get("data", [])

    # Query addresses table
    addresses_query = f"SELECT * FROM {chain_name}.erp_vendors.vendor_addresses LIMIT 1000"
    result = execute_sql_query(addresses_query, tool_context)

    if result.get("status") == "success":
        chain_data["addresses"] = result.get("data", [])

    return chain_data if chain_data else None


def _load_masterdata(tool_context: ToolContext) -> dict | None:
    """Load masterdata vendor records."""
    databricks_host = os.getenv("DATABRICKS_HOST")

    if not databricks_host:
        # Simulation mode
        return _simulate_masterdata()

    # Real Databricks mode
    from rlm_adk.tools.databricks_repl import execute_sql_query

    masterdata = {}

    # Query golden records
    golden_query = "SELECT * FROM masterdata_vendors.golden_records.vendors LIMIT 1000"
    result = execute_sql_query(golden_query, tool_context)

    if result.get("status") == "success":
        masterdata["golden_records"] = result.get("data", [])

    # Query mappings
    mappings_query = "SELECT * FROM masterdata_vendors.mappings.vendor_mappings LIMIT 1000"
    result = execute_sql_query(mappings_query, tool_context)

    if result.get("status") == "success":
        masterdata["mappings"] = result.get("data", [])

    return masterdata if masterdata else None


def _simulate_chain_data(chain_name: str) -> dict:
    """Simulate hospital chain vendor data for development."""
    chain_configs = {
        "hospital_chain_alpha": {
            "vendors": [
                {"vendor_id": "ALPHA-V001", "vendor_name": "MedSupply Corp", "tax_id": "12-3456789", "status": "ACTIVE"},
                {"vendor_id": "ALPHA-V002", "vendor_name": "HealthEquip Inc", "tax_id": "98-7654321", "status": "ACTIVE"},
                {"vendor_id": "ALPHA-V003", "vendor_name": "Surgical Solutions LLC", "tax_id": "55-1234567", "status": "ACTIVE"},
                {"vendor_id": "ALPHA-V004", "vendor_name": "PharmaCare Plus", "tax_id": "33-9876543", "status": "ACTIVE"},
                {"vendor_id": "ALPHA-V005", "vendor_name": "MedDevice Pro", "tax_id": "44-1122334", "status": "INACTIVE"},
            ],
            "addresses": [
                {"vendor_id": "ALPHA-V001", "address": "123 Medical Dr, Boston, MA 02101", "type": "PRIMARY"},
                {"vendor_id": "ALPHA-V002", "address": "456 Health Ave, Chicago, IL 60601", "type": "PRIMARY"},
                {"vendor_id": "ALPHA-V003", "address": "789 Surgery Ln, Houston, TX 77001", "type": "PRIMARY"},
            ],
        },
        "hospital_chain_beta": {
            "vendors": [
                {"vendor_id": "BETA-V001", "vendor_name": "MedSupply Corporation", "tax_id": "12-3456789", "status": "ACTIVE"},
                {"vendor_id": "BETA-V002", "vendor_name": "Health Equipment Inc.", "tax_id": "98-7654321", "status": "ACTIVE"},
                {"vendor_id": "BETA-V003", "vendor_name": "BioMed Supplies", "tax_id": "77-5544332", "status": "ACTIVE"},
                {"vendor_id": "BETA-V004", "vendor_name": "Pharma Care+", "tax_id": "33-9876543", "status": "ACTIVE"},
            ],
            "addresses": [
                {"vendor_id": "BETA-V001", "address": "123 Medical Drive, Boston, MA 02101", "type": "MAIN"},
                {"vendor_id": "BETA-V002", "address": "456 Healthcare Ave, Chicago, IL 60602", "type": "MAIN"},
            ],
        },
        "hospital_chain_gamma": {
            "vendors": [
                {"vendor_id": "GAMMA-V001", "vendor_name": "Med Supply Corp.", "tax_id": "12-3456789", "status": "ACTIVE"},
                {"vendor_id": "GAMMA-V002", "vendor_name": "HealthEquip", "tax_id": "98-7654321", "status": "ACTIVE"},
                {"vendor_id": "GAMMA-V003", "vendor_name": "Surgical Sols", "tax_id": "55-1234567", "status": "ACTIVE"},
                {"vendor_id": "GAMMA-V004", "vendor_name": "NovaMed Systems", "tax_id": "66-7788990", "status": "ACTIVE"},
            ],
            "addresses": [
                {"vendor_id": "GAMMA-V001", "address": "123 Medical Dr, Boston, Massachusetts 02101", "type": "HQ"},
                {"vendor_id": "GAMMA-V003", "address": "789 Surgery Lane, Houston, TX 77001", "type": "HQ"},
            ],
        },
    }

    return chain_configs.get(chain_name, {
        "vendors": [
            {"vendor_id": f"{chain_name.upper()[:5]}-V001", "vendor_name": "Unknown Vendor", "tax_id": "00-0000000", "status": "UNKNOWN"}
        ],
        "addresses": [],
    })


def _simulate_masterdata() -> dict:
    """Simulate masterdata vendor records for development."""
    return {
        "golden_records": [
            {
                "masterdata_id": "MD-001",
                "canonical_name": "MedSupply Corporation",
                "tax_id": "12-3456789",
                "duns_number": "123456789",
                "primary_address": "123 Medical Drive, Boston, MA 02101",
                "verified_at": "2024-06-15",
            },
            {
                "masterdata_id": "MD-002",
                "canonical_name": "HealthEquip Inc.",
                "tax_id": "98-7654321",
                "duns_number": "987654321",
                "primary_address": "456 Health Avenue, Chicago, IL 60601",
                "verified_at": "2024-05-20",
            },
            {
                "masterdata_id": "MD-003",
                "canonical_name": "Surgical Solutions LLC",
                "tax_id": "55-1234567",
                "duns_number": "551234567",
                "primary_address": "789 Surgery Lane, Houston, TX 77001",
                "verified_at": "2024-04-10",
            },
        ],
        "mappings": [
            {"mapping_id": "MAP-001", "source_vendor_id": "ALPHA-V001", "source_chain": "hospital_chain_alpha", "masterdata_id": "MD-001", "confidence": 0.98},
            {"mapping_id": "MAP-002", "source_vendor_id": "BETA-V001", "source_chain": "hospital_chain_beta", "masterdata_id": "MD-001", "confidence": 0.95},
            {"mapping_id": "MAP-003", "source_vendor_id": "ALPHA-V002", "source_chain": "hospital_chain_alpha", "masterdata_id": "MD-002", "confidence": 0.97},
        ],
    }


def load_custom_context(
    data: Any,
    description: str,
    tool_context: ToolContext,
) -> dict:
    """Load custom data into the RLM context.

    A general-purpose context loader for any data type. Use this when
    you have custom data that doesn't fit the vendor data pattern.

    Args:
        data: Any data to load (list, dict, string, etc.)
        description: Human-readable description of the data.

    Returns:
        dict: Loading results.

    Example:
        Load custom analysis results:
        ```
        load_custom_context(
            data={"analysis_results": [...], "metadata": {...}},
            description="Pre-computed analysis results for aggregation"
        )
        ```
    """
    return rlm_load_context(
        context_data=data,
        context_description=description,
        tool_context=tool_context,
    )


def _load_query_results_native(
    sql_query: str,
    description: str,
    tool_context: ToolContext,
) -> dict:
    """Execute query and load as DataFrame in native mode."""
    spark = get_spark_session()

    try:
        df = spark.sql(sql_query)
        row_count = df.count()
        columns = df.columns

        # Create temp view
        df.createOrReplaceTempView("rlm_query_context")

        load_result = rlm_load_context(
            context_data=df,
            context_description=description,
            tool_context=tool_context,
        )

        return {
            "status": "success",
            "rows_loaded": row_count,
            "columns": columns,
            "description": description,
            "context_type": "spark_dataframe",
            "temp_view": "rlm_query_context",
            "load_result": load_result,
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Query failed: {str(e)}",
            "query": sql_query,
        }


def load_query_results_to_context(
    sql_query: str,
    description: str,
    tool_context: ToolContext,
) -> dict:
    """Execute a SQL query and load results into RLM context.

    Combines SQL execution with context loading for convenience.

    Args:
        sql_query: SQL query to execute against Databricks.
        description: Description of the query results.

    Returns:
        dict: Query and loading results.

    Example:
        Load specific vendor subset:
        ```
        load_query_results_to_context(
            sql_query="SELECT * FROM hospital_chain_alpha.erp_vendors.vendors WHERE status='ACTIVE'",
            description="Active vendors from Alpha hospital chain"
        )
        ```
    """
    # Check for native mode
    if get_execution_mode() == "native":
        return _load_query_results_native(sql_query, description, tool_context)

    # Existing local mode implementation
    from rlm_adk.tools.databricks_repl import execute_sql_query

    # Execute the query
    query_result = execute_sql_query(sql_query, tool_context)

    if query_result.get("status") != "success":
        return {
            "status": "error",
            "error_message": f"Query failed: {query_result.get('error_message', 'Unknown error')}",
            "query": sql_query,
        }

    # Load results into context
    data = query_result.get("data", [])
    columns = query_result.get("columns", [])

    # Convert to list of dicts if we have column names
    if columns and data:
        structured_data = [
            dict(zip(columns, row, strict=False)) for row in data
        ]
    else:
        structured_data = data

    load_result = rlm_load_context(
        context_data=structured_data,
        context_description=description,
        tool_context=tool_context,
    )

    return {
        "status": "success",
        "rows_loaded": len(structured_data),
        "columns": columns,
        "description": description,
        "load_result": load_result,
    }
