"""Databricks Job entrypoint for RLM-ADK vendor resolution.

This module provides a clean entrypoint for running the RLM-ADK root_agent
as a Databricks Workflow Job. It can be invoked via:

1. Console script: `rlm-adk-job` (after pip install)
2. Python wheel task in Databricks Jobs
3. Direct execution: `python -m rlm_adk.entrypoint`

Environment Variables:
    RLM_QUERY: The query/task for the agent to execute
    RLM_HOSPITAL_CHAINS: Comma-separated list of hospital chain catalogs
    RLM_INCLUDE_MASTERDATA: Whether to include masterdata (true/false)
    GOOGLE_API_KEY: API key for Gemini LLM
    RLM_EXECUTION_MODE: Force execution mode (native/local)

Example Databricks Job Configuration:
    {
        "tasks": [{
            "task_key": "vendor_resolution",
            "python_wheel_task": {
                "package_name": "rlm",
                "entry_point": "rlm-adk-job",
                "parameters": []
            },
            "libraries": [{"pypi": {"package": "rlm[rlm-adk]"}}],
            "new_cluster": {...}
        }]
    }
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime


def main() -> int:
    """Run the RLM-ADK root_agent as a Databricks Job.

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    print("=" * 60)
    print("RLM-ADK Job Starting")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    # Get runtime info
    from rlm_adk.runtime import get_runtime_info
    runtime_info = get_runtime_info()
    print("\nRuntime Info:")
    for key, value in runtime_info.items():
        print(f"  {key}: {value}")

    # Parse configuration from environment
    query = os.getenv(
        "RLM_QUERY",
        "Find duplicate vendors across all hospital chains based on tax ID and name similarity"
    )

    hospital_chains_str = os.getenv(
        "RLM_HOSPITAL_CHAINS",
        "hospital_chain_alpha,hospital_chain_beta,hospital_chain_gamma"
    )
    hospital_chains = [c.strip() for c in hospital_chains_str.split(",") if c.strip()]

    include_masterdata = os.getenv("RLM_INCLUDE_MASTERDATA", "true").lower() == "true"

    print("\nJob Configuration:")
    print(f"  Query: {query}")
    print(f"  Hospital Chains: {hospital_chains}")
    print(f"  Include Masterdata: {include_masterdata}")

    # Validate environment
    if not os.getenv("GOOGLE_API_KEY"):
        print("\nWARNING: GOOGLE_API_KEY not set. LLM calls will use simulation mode.")

    try:
        # Import the agent and runner
        from rlm_adk.tools.context_loader import load_vendor_data_to_context
        from rlm_adk.tools.rlm_tools import rlm_query_context

        print("\n" + "-" * 60)
        print("Phase 1: Loading Context")
        print("-" * 60)

        # Create a mock tool context for standalone execution
        from unittest.mock import MagicMock
        tool_context = MagicMock()
        tool_context.state = {}

        # Load vendor data
        load_result = load_vendor_data_to_context(
            hospital_chains=hospital_chains,
            include_masterdata=include_masterdata,
            tool_context=tool_context,
        )

        print("\nContext Load Result:")
        print(f"  Status: {load_result.get('status')}")
        print(f"  Total Vendors: {load_result.get('total_vendors')}")
        print(f"  Chains Loaded: {load_result.get('chains_loaded')}")

        if load_result.get("status") != "success":
            print(f"\nERROR: Failed to load context: {load_result.get('error_message')}")
            return 1

        print("\n" + "-" * 60)
        print("Phase 2: Executing Query")
        print("-" * 60)
        print(f"\nQuery: {query}")

        # Determine strategy based on execution mode
        strategy = "spark_sql" if runtime_info["execution_mode"] == "native" else "chunk_and_aggregate"
        print(f"Strategy: {strategy}")

        # Execute the query using RLM decomposition
        query_result = rlm_query_context(
            query=query,
            strategy=strategy,
            tool_context=tool_context,
        )

        print("\n" + "-" * 60)
        print("Phase 3: Results")
        print("-" * 60)

        print("\nQuery Result:")
        print(f"  Status: {query_result.get('status')}")
        print(f"  Iterations: {query_result.get('iterations')}")
        print(f"  LLM Calls: {query_result.get('llm_calls')}")
        print(f"\nAnswer:\n{query_result.get('answer', 'No answer generated')}")

        # Write results to output (for Databricks Job output)
        result_summary = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "hospital_chains": hospital_chains,
            "total_vendors": load_result.get("total_vendors"),
            "iterations": query_result.get("iterations"),
            "llm_calls": query_result.get("llm_calls"),
            "answer": query_result.get("answer"),
        }

        print("\n" + "=" * 60)
        print("RLM-ADK Job Completed Successfully")
        print("=" * 60)

        # Output JSON result for job output capture
        print(f"\n__RESULT_JSON__: {json.dumps(result_summary)}")

        return 0

    except Exception as e:
        print(f"\nERROR: Job failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cli():
    """CLI entrypoint for console_scripts."""
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
