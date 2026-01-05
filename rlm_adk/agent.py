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

from google.adk.agents import Agent, LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools import FunctionTool

from rlm_adk.agents.erp_analyzer import (
    alpha_erp_analyzer,
    beta_erp_analyzer,
    gamma_erp_analyzer,
)
from rlm_adk.agents.vendor_matcher import vendor_matcher_agent
from rlm_adk.agents.view_generator import view_generator_agent

# Import RLM workflow components
from rlm_adk.agents.rlm_loop import (
    make_rlm_completion_workflow,
    make_rlm_iteration_loop,
)
from rlm_adk.callbacks import (
    before_model_callback,
    after_model_callback,
    on_model_error_callback,
)
from rlm_adk.prompts import ROOT_AGENT_INSTRUCTION

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


def _clear_parent(agent):
    if agent is not None and hasattr(agent, "parent_agent"):
        agent.parent_agent = None
    return agent


# =========================================================================
# RLM Analyst Agent: Recursive Decomposition for Large Data
# =========================================================================

rlm_analyst_agent = Agent(
    name="rlm_analyst",
    model="gemini-3.0-flash",
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
        FunctionTool(rlm_execute_code),
        FunctionTool(rlm_load_context),
        FunctionTool(rlm_query_context),
        FunctionTool(rlm_get_session_state),
        FunctionTool(rlm_clear_session),
        # Context loaders
        FunctionTool(load_vendor_data_to_context),
        FunctionTool(load_custom_context),
        FunctionTool(load_query_results_to_context),
        # Data exploration (for context building)
        FunctionTool(list_catalogs),
        FunctionTool(list_schemas),
        FunctionTool(list_tables),
        FunctionTool(read_table_sample),
        FunctionTool(execute_sql_query),
    ],
    output_key="rlm_analysis_results",
)

# =========================================================================
# Parallel Agent: Concurrent ERP Analysis
# =========================================================================

parallel_erp_analysis = ParallelAgent(
    name="parallel_erp_analysis",
    description="""Concurrent analysis of vendor data across all hospital chain ERP systems.
    Runs analyzers for Alpha, Beta, and Gamma hospital chains simultaneously.""",
    sub_agents=[
        _clear_parent(alpha_erp_analyzer),
        _clear_parent(beta_erp_analyzer),
        _clear_parent(gamma_erp_analyzer),
    ],
)

# =========================================================================
# Sequential Agent: Complete Vendor Resolution Pipeline
# =========================================================================

vendor_resolution_pipeline = SequentialAgent(
    name="vendor_resolution_pipeline",
    description="""End-to-end vendor resolution workflow:
    1. Parallel ERP analysis across all hospital chains
    2. Vendor matching and masterdata resolution
    3. Analytical view generation""",
    sub_agents=[
        _clear_parent(parallel_erp_analysis),   # Step 1: Analyze all ERPs concurrently
        _clear_parent(vendor_matcher_agent),     # Step 2: Match and resolve vendors
        _clear_parent(view_generator_agent),     # Step 3: Create unified views
    ],
)

# =========================================================================
# RLM Workflow Agents: Full Recursive Decomposition Pipeline
# =========================================================================

# Full RLM completion workflow (nested LoopAgent)
rlm_workflow = make_rlm_completion_workflow(max_iterations=10)

# Direct access to just the iteration loop
rlm_loop = make_rlm_iteration_loop(max_iterations=10)

# =========================================================================
# Root Agent: Data Scientist Interface with RLM Integration
# =========================================================================

root_agent = LlmAgent(
    name="rlm_data_scientist",
    model="gemini-3.0",
    description="""Healthcare data scientist with full RLM recursive decomposition capabilities.
    Can programmatically decompose large vendor resolution problems using sub-LM calls.""",
    instruction=ROOT_AGENT_INSTRUCTION,  # From rlm_adk/prompts.py
    tools=[
        # Direct RLM tools for simple cases
        FunctionTool(rlm_execute_code),
        FunctionTool(rlm_load_context),
        FunctionTool(rlm_query_context),
        FunctionTool(rlm_get_session_state),
        FunctionTool(rlm_clear_session),
        # Context loaders
        FunctionTool(load_vendor_data_to_context),
        FunctionTool(load_custom_context),
        FunctionTool(load_query_results_to_context),
        # Data exploration tools
        FunctionTool(list_catalogs),
        FunctionTool(list_schemas),
        FunctionTool(list_tables),
        FunctionTool(list_volumes),
        FunctionTool(get_volume_metadata),
        FunctionTool(read_table_sample),
        # Databricks REPL tools
        FunctionTool(execute_sql_query),
        FunctionTool(execute_python_code),
        FunctionTool(get_repl_session_state),
        # Vendor resolution tools
        FunctionTool(find_similar_vendors),
        FunctionTool(search_vendor_by_attributes),
        FunctionTool(get_masterdata_vendor),
        FunctionTool(resolve_vendor_to_masterdata),
        FunctionTool(create_vendor_mapping),
        # View creation
        FunctionTool(create_view),
    ],
    sub_agents=[
        _clear_parent(rlm_workflow),                # Full RLM workflow (recommended)
        _clear_parent(rlm_loop),                    # Just the iteration loop
        _clear_parent(rlm_analyst_agent),          # Legacy RLM analyst
        _clear_parent(vendor_resolution_pipeline), # Standard pipeline
        _clear_parent(parallel_erp_analysis),      # Parallel ERP analysis
        _clear_parent(vendor_matcher_agent),        # Direct vendor matching
        _clear_parent(view_generator_agent),        # Direct view generation
    ],
    # Root agent callbacks for metrics
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    on_model_error_callback=on_model_error_callback,
)


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
]
