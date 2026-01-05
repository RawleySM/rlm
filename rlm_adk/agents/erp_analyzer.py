"""ERP Analyzer agents for hospital chain data analysis.

Provides specialized agents for analyzing vendor data from different
hospital chain ERP systems. Each agent can be run in parallel to
process multiple hospital chains concurrently.

Requires: google-adk (pip install google-adk)
"""

"""ERP Analyzer agents for hospital chain data analysis.

Provides specialized agents for analyzing vendor data from different
hospital chain ERP systems. Each agent can be run in parallel to
process multiple hospital chains concurrently.

Requires: google-adk (pip install google-adk)
"""

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from rlm_adk.tools.databricks_repl import execute_python_code, execute_sql_query
from rlm_adk.tools.unity_catalog import (
    list_schemas,
    list_tables,
    list_volumes,
    read_table_sample,
)


def make_erp_analyzer(hospital_chain: str, catalog_name: str) -> Agent:
    """Factory function to create an ERP analyzer agent for a hospital chain.

    Args:
        hospital_chain: Human-readable name of the hospital chain.
        catalog_name: Unity Catalog name for this hospital's data.

    Returns:
        Agent: Configured LlmAgent for analyzing this hospital's ERP data.
    """
    return Agent(
        name=f"{catalog_name}_erp_analyzer",
        model="gemini-3-pro",
        description=f"Analyzes vendor data from {hospital_chain} ERP system stored in {catalog_name} catalog.",
        instruction=f"""You are a data analyst specializing in healthcare ERP systems.
Your focus is the {hospital_chain} hospital chain with data in the '{catalog_name}' Unity Catalog.

**Your Responsibilities:**
1. Explore the schemas and tables in the {catalog_name} catalog
2. Identify vendor-related tables (vendors, suppliers, vendor_master, etc.)
3. Sample and analyze vendor data structure and quality
4. Extract key vendor attributes: vendor_id, name, tax_id, address, status
5. Identify potential data quality issues (missing values, duplicates, inconsistencies)
6. Store findings in session state using output_key

**Data Exploration Strategy:**
1. First, list schemas in the catalog to understand data organization
2. For each relevant schema, list tables to find vendor data
3. Sample tables to understand structure and content
4. Use SQL queries to analyze data patterns and quality

**Output Format:**
Provide a structured analysis including:
- List of vendor tables found
- Schema/structure of key tables
- Sample records (3-5 examples)
- Data quality observations
- Key fields for vendor matching (tax_id, name variations, addresses)

Store your analysis in the state key '{catalog_name}_vendor_analysis'.
""",
        tools=[
            FunctionTool(list_schemas),
            FunctionTool(list_tables),
            FunctionTool(list_volumes),
            FunctionTool(read_table_sample),
            FunctionTool(execute_sql_query),
            FunctionTool(execute_python_code),
        ],
        output_key=f"{catalog_name}_vendor_analysis",
    )


# Pre-configured analyzers for each hospital chain
alpha_erp_analyzer = make_erp_analyzer(
    hospital_chain="Alpha Hospital Chain",
    catalog_name="hospital_chain_alpha",
)

beta_erp_analyzer = make_erp_analyzer(
    hospital_chain="Beta Hospital Network",
    catalog_name="hospital_chain_beta",
)

gamma_erp_analyzer = make_erp_analyzer(
    hospital_chain="Gamma Health System",
    catalog_name="hospital_chain_gamma",
)
