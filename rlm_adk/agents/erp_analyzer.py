"""ERP Analyzer agents for hospital chain data analysis.

Provides specialized agents for analyzing vendor data from different
hospital chain ERP systems. Each agent can be run in parallel to
process multiple hospital chains concurrently.

Requires: google-adk (pip install google-adk)
"""

from rlm_adk._compat import ADK_AVAILABLE, check_adk_available


def make_erp_analyzer(hospital_chain: str, catalog_name: str):
    """Factory function to create an ERP analyzer agent for a hospital chain.

    Args:
        hospital_chain: Human-readable name of the hospital chain.
        catalog_name: Unity Catalog name for this hospital's data.

    Returns:
        Agent: Configured LlmAgent for analyzing this hospital's ERP data.

    Raises:
        ImportError: If google-adk is not installed.
    """
    check_adk_available()

    from google.adk.agents import Agent

    from rlm_adk.tools.databricks_repl import execute_python_code, execute_sql_query
    from rlm_adk.tools.unity_catalog import (
        list_schemas,
        list_tables,
        list_volumes,
        read_table_sample,
    )

    return Agent(
        name=f"{catalog_name}_erp_analyzer",
        model="gemini-2.0-flash",
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
            list_schemas,
            list_tables,
            list_volumes,
            read_table_sample,
            execute_sql_query,
            execute_python_code,
        ],
        output_key=f"{catalog_name}_vendor_analysis",
    )


class _LazyERPAnalyzer:
    """Lazy proxy for ERP analyzer agents."""

    def __init__(self, hospital_chain: str, catalog_name: str):
        self._hospital_chain = hospital_chain
        self._catalog_name = catalog_name
        self._agent = None

    def _ensure_loaded(self):
        if self._agent is None:
            self._agent = make_erp_analyzer(self._hospital_chain, self._catalog_name)
        return self._agent

    @property
    def name(self):
        return f"{self._catalog_name}_erp_analyzer"

    @property
    def output_key(self):
        return f"{self._catalog_name}_vendor_analysis"

    def __getattr__(self, name):
        return getattr(self._ensure_loaded(), name)

    def __repr__(self):
        if ADK_AVAILABLE:
            return repr(self._ensure_loaded())
        return f"<LazyERPAnalyzer({self._catalog_name}) - google-adk not installed>"


# Pre-configured analyzers for each hospital chain (lazy loaded)
alpha_erp_analyzer = _LazyERPAnalyzer(
    hospital_chain="Alpha Hospital Chain",
    catalog_name="hospital_chain_alpha",
)

beta_erp_analyzer = _LazyERPAnalyzer(
    hospital_chain="Beta Hospital Network",
    catalog_name="hospital_chain_beta",
)

gamma_erp_analyzer = _LazyERPAnalyzer(
    hospital_chain="Gamma Health System",
    catalog_name="hospital_chain_gamma",
)
