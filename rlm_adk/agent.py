"""RLM-ADK Agent: Data Scientist for Healthcare Vendor Management.

This module defines the root_agent and orchestration workflows for
analyzing hospital ERP data across Unity Catalog volumes and resolving
vendor instances to masterdata golden records.

Architecture:
- ParallelAgent: Concurrent analysis of multiple hospital chain ERPs
- SequentialAgent: Orchestrated workflow from analysis to view creation
- Root Agent: Interactive data scientist interface

Usage:
    from rlm_adk import root_agent
    # Use with ADK Runner or adk web/run commands

Requirements:
    pip install google-adk  # or: pip install rlm[rlm-adk]
"""

from rlm_adk._compat import ADK_AVAILABLE, check_adk_available

# Lazy initialization - agents are only created when accessed
_parallel_erp_analysis = None
_vendor_resolution_pipeline = None
_root_agent = None


def _get_agents():
    """Lazily initialize ADK agents when first accessed."""
    global _parallel_erp_analysis, _vendor_resolution_pipeline, _root_agent

    if _root_agent is not None:
        return _parallel_erp_analysis, _vendor_resolution_pipeline, _root_agent

    # Check ADK is available before importing
    check_adk_available()

    from google.adk.agents import Agent, ParallelAgent, SequentialAgent

    from rlm_adk.agents.erp_analyzer import (
        alpha_erp_analyzer,
        beta_erp_analyzer,
        gamma_erp_analyzer,
    )
    from rlm_adk.agents.vendor_matcher import vendor_matcher_agent
    from rlm_adk.agents.view_generator import view_generator_agent
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

    # =========================================================================
    # Parallel Agent: Concurrent ERP Analysis
    # =========================================================================

    _parallel_erp_analysis = ParallelAgent(
        name="parallel_erp_analysis",
        description="""Concurrent analysis of vendor data across all hospital chain ERP systems.
        Runs analyzers for Alpha, Beta, and Gamma hospital chains simultaneously.""",
        sub_agents=[
            alpha_erp_analyzer,
            beta_erp_analyzer,
            gamma_erp_analyzer,
        ],
    )

    # =========================================================================
    # Sequential Agent: Complete Vendor Resolution Pipeline
    # =========================================================================

    _vendor_resolution_pipeline = SequentialAgent(
        name="vendor_resolution_pipeline",
        description="""End-to-end vendor resolution workflow:
        1. Parallel ERP analysis across all hospital chains
        2. Vendor matching and masterdata resolution
        3. Analytical view generation""",
        sub_agents=[
            _parallel_erp_analysis,   # Step 1: Analyze all ERPs concurrently
            vendor_matcher_agent,     # Step 2: Match and resolve vendors
            view_generator_agent,     # Step 3: Create unified views
        ],
    )

    # =========================================================================
    # Root Agent: Data Scientist Interface
    # =========================================================================

    _root_agent = Agent(
        name="rlm_data_scientist",
        model="gemini-2.0-flash",
        description="""Healthcare data scientist specializing in vendor master data management.
        Analyzes hospital ERP databases, resolves vendor duplicates, and creates unified analytics.""",
        instruction="""You are an expert data scientist specializing in healthcare vendor master data management.

**Your Environment:**
You work with a Databricks workspace containing Unity Catalog volumes with data from:
- Multiple hospital chain ERP systems (Alpha, Beta, Gamma)
- A masterdata vendor database for golden record management

**Your Capabilities:**

1. **Data Exploration:**
   - List and explore Unity Catalogs, schemas, tables, and volumes
   - Sample and analyze table data
   - Execute SQL queries and Python code in the Databricks REPL

2. **Vendor Analysis:**
   - Find similar vendors across hospital chains
   - Search vendors by attributes (Tax ID, DUNS, address)
   - Identify potential duplicate vendor records

3. **Masterdata Management:**
   - Resolve vendor records to masterdata golden records
   - Create new masterdata vendor entities
   - Build vendor mappings with confidence scores

4. **Analytical Views:**
   - Create views joining data across hospital chains
   - Build unified vendor reporting views
   - Generate data quality reports

**Workflow Options:**

A. **Interactive Analysis:** Answer questions, explore data, run ad-hoc queries
   - Use tools directly for specific tasks
   - Great for: Quick lookups, data quality checks, custom analysis

B. **Full Pipeline Execution:** Run the complete vendor resolution workflow
   - Delegate to `vendor_resolution_pipeline` for end-to-end processing
   - Great for: Initial setup, batch processing, comprehensive analysis

**Communication Style:**
- Explain your analysis approach before executing
- Provide clear summaries of findings
- Highlight data quality issues and recommendations
- Use tables and structured output for complex results

**Example Interactions:**

User: "Show me the vendor tables in hospital_chain_alpha"
Action: Use list_schemas and list_tables to explore the catalog

User: "Find all vendors similar to 'MedSupply Corp'"
Action: Use find_similar_vendors with appropriate threshold

User: "Run the full vendor resolution pipeline"
Action: Delegate to vendor_resolution_pipeline sub-agent

User: "Create a view showing unified vendor data"
Action: Use create_view or delegate to view_generator_agent

**State Keys Available:**
After pipeline execution, these state keys contain results:
- `hospital_chain_alpha_vendor_analysis`: Alpha chain analysis
- `hospital_chain_beta_vendor_analysis`: Beta chain analysis
- `hospital_chain_gamma_vendor_analysis`: Gamma chain analysis
- `vendor_resolution_results`: Matching and resolution results
- `created_views`: Generated view definitions

Always be helpful, thorough, and explain your reasoning!
""",
        tools=[
            # Data exploration tools
            list_catalogs,
            list_schemas,
            list_tables,
            list_volumes,
            get_volume_metadata,
            read_table_sample,
            # REPL tools
            execute_sql_query,
            execute_python_code,
            get_repl_session_state,
            # Vendor resolution tools
            find_similar_vendors,
            search_vendor_by_attributes,
            get_masterdata_vendor,
            resolve_vendor_to_masterdata,
            create_vendor_mapping,
            # View creation
            create_view,
        ],
        sub_agents=[
            _vendor_resolution_pipeline,   # Full pipeline delegation
            _parallel_erp_analysis,        # Just the parallel analysis
            vendor_matcher_agent,          # Just vendor matching
            view_generator_agent,          # Just view generation
        ],
    )

    return _parallel_erp_analysis, _vendor_resolution_pipeline, _root_agent


class _LazyAgent:
    """Lazy proxy for ADK agents that initializes on first access."""

    def __init__(self, name: str, index: int):
        self._name = name
        self._index = index
        self._agent = None

    def _ensure_loaded(self):
        if self._agent is None:
            agents = _get_agents()
            self._agent = agents[self._index]
        return self._agent

    def __getattr__(self, name):
        return getattr(self._ensure_loaded(), name)

    def __repr__(self):
        if ADK_AVAILABLE:
            return repr(self._ensure_loaded())
        return f"<LazyAgent({self._name}) - google-adk not installed>"


# Create lazy proxies for the agents
parallel_erp_analysis = _LazyAgent("parallel_erp_analysis", 0)
vendor_resolution_pipeline = _LazyAgent("vendor_resolution_pipeline", 1)
root_agent = _LazyAgent("root_agent", 2)


# Also export the sub-agents when accessed through this module
def __getattr__(name):
    """Module-level getattr for lazy agent access."""
    if name == "vendor_matcher_agent":
        check_adk_available()
        from rlm_adk.agents.vendor_matcher import vendor_matcher_agent
        return vendor_matcher_agent
    elif name == "view_generator_agent":
        check_adk_available()
        from rlm_adk.agents.view_generator import view_generator_agent
        return view_generator_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# Convenience exports
# =============================================================================

__all__ = [
    "root_agent",
    "vendor_resolution_pipeline",
    "parallel_erp_analysis",
    "vendor_matcher_agent",
    "view_generator_agent",
    "ADK_AVAILABLE",
]
