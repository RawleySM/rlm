"""RLM-ADK Agent: Data Scientist for Healthcare Vendor Management.

This module defines the root_agent and orchestration workflows for
analyzing hospital ERP data across Unity Catalog volumes and resolving
vendor instances to masterdata golden records.

**RLM Integration:**
This agent integrates the Recursive Language Model (RLM) paradigm with
Google ADK. Key RLM principles honored:

1. Context Offloading - Large datasets loaded into REPL `context` variable
2. llm_query() - Recursive sub-LM calls from within code execution
3. llm_query_batched() - Concurrent sub-LM calls for parallel decomposition
4. Iterative Execution - Code blocks executed until FINAL answer

The agent can programmatically decompose large problems (like vendor
resolution across millions of records) by spawning sub-LM calls.

Architecture:
- RLM REPL with llm_query support for recursive decomposition
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
_rlm_analyst_agent = None


def _get_agents():
    """Lazily initialize ADK agents when first accessed."""
    global _parallel_erp_analysis, _vendor_resolution_pipeline, _root_agent, _rlm_analyst_agent

    if _root_agent is not None:
        return _parallel_erp_analysis, _vendor_resolution_pipeline, _root_agent, _rlm_analyst_agent

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

    # Import all tools
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

    # =========================================================================
    # RLM Analyst Agent: Recursive Decomposition for Large Data
    # =========================================================================

    _rlm_analyst_agent = Agent(
        name="rlm_analyst",
        model="gemini-2.0-flash",
        description="""RLM-powered analyst that uses recursive decomposition to analyze
        large datasets. Uses llm_query() and llm_query_batched() for sub-LM calls.""",
        instruction="""You are an RLM (Recursive Language Model) analyst. You have a unique
capability: you can execute Python code that spawns sub-LM calls to analyze large contexts.

**Your RLM Capabilities:**

1. **Context Loading**: Use `load_vendor_data_to_context` to load large datasets
   into the REPL's `context` variable.

2. **Recursive Analysis**: Use `rlm_execute_code` to run Python code with access to:
   - `context`: The loaded data
   - `llm_query(prompt)`: Make a sub-LM call
   - `llm_query_batched(prompts)`: Make concurrent sub-LM calls

3. **Decomposition Strategies**: Use `rlm_query_context` for pre-built strategies:
   - `chunk_and_aggregate`: Split data, analyze chunks, combine results
   - `iterative`: Process sequentially with state tracking
   - `map_reduce`: Map analysis, then reduce to answer
   - `hierarchical`: Build hierarchical summaries

**Example: Analyzing Vendor Duplicates**

```python
# Step 1: Load vendor data from all hospital chains
load_vendor_data_to_context(
    hospital_chains=["hospital_chain_alpha", "hospital_chain_beta", "hospital_chain_gamma"],
    include_masterdata=True
)

# Step 2: Use RLM recursive decomposition to find duplicates
rlm_execute_code('''
# Context now contains all vendor data from 3 chains
# Analyze each chain with sub-LM calls
results = []
for chain_name, chain_data in context.items():
    if chain_name == "masterdata":
        continue
    vendors = chain_data.get("vendors", [])
    # Use llm_query to analyze this chain's vendors
    analysis = llm_query(f"Analyze these vendors for duplicates: {vendors}")
    results.append(f"{chain_name}: {analysis}")
    print(f"Analyzed {chain_name}: {len(vendors)} vendors")

# Aggregate with final sub-LM call
final_answer = llm_query(f"Combine duplicate findings: {results}")
print(f"FINAL: {final_answer}")
''')
```

**When to Use RLM Pattern:**
- Large datasets that need decomposition (thousands of records)
- Complex analysis requiring multiple reasoning steps
- Cross-chain vendor matching with fuzzy logic
- When you need to aggregate insights from many sources

Always explain your decomposition strategy before executing!
""",
        tools=[
            # RLM REPL tools (primary)
            rlm_execute_code,
            rlm_load_context,
            rlm_query_context,
            rlm_get_session_state,
            rlm_clear_session,
            # Context loaders
            load_vendor_data_to_context,
            load_custom_context,
            load_query_results_to_context,
            # Data exploration (for context building)
            list_catalogs,
            list_schemas,
            list_tables,
            read_table_sample,
            execute_sql_query,
        ],
        output_key="rlm_analysis_results",
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
    # Root Agent: Data Scientist Interface with RLM Integration
    # =========================================================================

    _root_agent = Agent(
        name="rlm_data_scientist",
        model="gemini-2.0-flash",
        description="""Healthcare data scientist with RLM (Recursive Language Model) capabilities.
        Can programmatically decompose large vendor resolution problems using sub-LM calls.""",
        instruction="""You are an expert data scientist with RLM (Recursive Language Model) capabilities
for healthcare vendor master data management.

**Your Environment:**
You work with a Databricks workspace containing Unity Catalog volumes with data from:
- Multiple hospital chain ERP systems (Alpha, Beta, Gamma)
- A masterdata vendor database for golden record management

**RLM CAPABILITIES (Key Differentiator):**

You have access to the RLM paradigm which allows you to:

1. **Offload Context**: Load large datasets into the REPL's `context` variable
   - Use `load_vendor_data_to_context` for vendor data
   - Use `load_query_results_to_context` for SQL results

2. **Recursive Sub-LM Calls**: Execute code with `llm_query()` access
   - `rlm_execute_code` - Run Python with `context`, `llm_query()`, `llm_query_batched()`
   - The LM can spawn sub-LM calls to decompose large problems

3. **Decomposition Strategies**:
   - `rlm_query_context` with strategy="chunk_and_aggregate" - Split and combine
   - `rlm_query_context` with strategy="map_reduce" - Map then reduce
   - `rlm_query_context` with strategy="iterative" - Sequential with state

**When to Use RLM Pattern:**
- Analyzing thousands of vendor records across chains
- Finding duplicates that require semantic understanding
- Complex aggregation across multiple data sources
- When simple tools aren't sufficient

**Example RLM Workflow:**
```
1. load_vendor_data_to_context(["hospital_chain_alpha", "hospital_chain_beta"], True)
2. rlm_query_context("Find all vendor duplicates based on tax ID and name similarity", "chunk_and_aggregate")
```

Or for custom logic:
```
1. load_vendor_data_to_context(...)
2. rlm_execute_code('''
   # Custom decomposition with llm_query
   for chain, data in context.items():
       analysis = llm_query(f"Analyze {chain} vendors: {data}")
       print(analysis)
   ''')
```

**Standard Capabilities (Non-RLM):**

1. **Data Exploration:**
   - List and explore Unity Catalogs, schemas, tables, and volumes
   - Sample and analyze table data
   - Execute SQL queries and Python code

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

**Workflow Options:**

A. **RLM Analysis (Recommended for Large Data):**
   - Load context → Execute with llm_query → Get decomposed results
   - Delegate to `rlm_analyst` for complex recursive analysis

B. **Direct Tool Use (Quick Queries):**
   - Use individual tools for specific tasks
   - Great for: Quick lookups, single-table queries

C. **Pipeline Execution (End-to-End):**
   - Delegate to `vendor_resolution_pipeline` for full workflow
   - Great for: Initial setup, comprehensive processing

**Communication Style:**
- Explain your approach before executing (especially RLM decomposition strategy)
- Provide clear summaries of findings
- Highlight when RLM pattern would be beneficial
- Use tables and structured output for complex results

Always be helpful, thorough, and leverage RLM for large-scale analysis!
""",
        tools=[
            # RLM REPL tools (PRIMARY - for recursive decomposition)
            rlm_execute_code,
            rlm_load_context,
            rlm_query_context,
            rlm_get_session_state,
            rlm_clear_session,
            # Context loaders
            load_vendor_data_to_context,
            load_custom_context,
            load_query_results_to_context,
            # Data exploration tools
            list_catalogs,
            list_schemas,
            list_tables,
            list_volumes,
            get_volume_metadata,
            read_table_sample,
            # Databricks REPL tools
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
            _rlm_analyst_agent,            # RLM-powered analysis
            _vendor_resolution_pipeline,   # Full pipeline delegation
            _parallel_erp_analysis,        # Just the parallel analysis
            vendor_matcher_agent,          # Just vendor matching
            view_generator_agent,          # Just view generation
        ],
    )

    return _parallel_erp_analysis, _vendor_resolution_pipeline, _root_agent, _rlm_analyst_agent


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
rlm_analyst_agent = _LazyAgent("rlm_analyst", 3)


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
    "rlm_analyst_agent",
    "vendor_resolution_pipeline",
    "parallel_erp_analysis",
    "vendor_matcher_agent",
    "view_generator_agent",
    "ADK_AVAILABLE",
]
