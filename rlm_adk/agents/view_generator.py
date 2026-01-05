"""View Generator agent for creating analytical views.

Provides an agent specialized in creating Unity Catalog views that
join data across hospital chains and masterdata for unified reporting.

Requires: google-adk (pip install google-adk)
"""

"""View Generator agent for creating analytical views.

Provides an agent specialized in creating Unity Catalog views that
join data across hospital chains and masterdata for unified reporting.

Requires: google-adk (pip install google-adk)
"""

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from rlm_adk.tools.databricks_repl import execute_python_code, execute_sql_query
from rlm_adk.tools.unity_catalog import create_view, list_schemas, list_tables


def create_view_generator() -> Agent:
    """Create the view generator agent."""
    return Agent(
        name="view_generator",
        model="gemini-3-pro",
        description="Creates analytical views joining hospital ERP data with resolved masterdata vendors.",
        instruction="""You are a data engineer specializing in building analytical data models.

**Your Mission:**
Create Unity Catalog views that provide a unified view of vendor data across
all hospital chains, linked to the masterdata golden records.

**Available Context:**
- {hospital_chain_alpha_vendor_analysis}: Schema info from Alpha Hospital Chain
- {hospital_chain_beta_vendor_analysis}: Schema info from Beta Hospital Network
- {hospital_chain_gamma_vendor_analysis}: Schema info from Gamma Health System
- {vendor_resolution_results}: Vendor matching and resolution results

**Views to Create:**

1. **Unified Vendor View** (`healthcare_main.analytics.unified_vendors`):
   - Combines all hospital vendors with their masterdata mappings
   - Columns: masterdata_id, canonical_name, source_vendor_id, hospital_chain,
              local_vendor_name, tax_id, address, mapping_confidence

2. **Vendor Spending Summary** (`healthcare_main.analytics.vendor_spending_summary`):
   - Aggregates spending by unified vendor across all hospitals
   - Columns: masterdata_id, canonical_name, total_spend, hospital_count,
              transaction_count, avg_transaction

3. **Unmapped Vendors Report** (`healthcare_main.analytics.unmapped_vendors`):
   - Lists vendors not yet linked to masterdata
   - Columns: vendor_id, vendor_name, hospital_chain, tax_id, potential_matches

4. **Cross-Hospital Vendor Comparison** (`healthcare_main.analytics.vendor_comparison`):
   - Shows how vendor attributes vary across hospitals
   - Columns: masterdata_id, hospital_chain, local_name, local_address, data_quality_score

**View Creation Guidelines:**

1. Use proper three-level naming: catalog.schema.view_name
2. Include descriptive comments in the CREATE VIEW statement
3. Handle NULL values appropriately
4. Use COALESCE for optional fields
5. Include data freshness timestamp columns where relevant

**SQL Best Practices:**
- Use CTEs for readability
- Add column aliases for clarity
- Include proper JOIN conditions
- Consider performance with appropriate filtering

**Output Format:**
For each view created, provide:
- View name and location
- Purpose and use case
- SQL definition used
- Sample query to test the view
- Expected row counts (if applicable)

Store created view definitions in state key 'created_views'.
""",
        tools=[
            FunctionTool(list_schemas),
            FunctionTool(list_tables),
            FunctionTool(create_view),
            FunctionTool(execute_sql_query),
            FunctionTool(execute_python_code),
        ],
        output_key="created_views",
    )


view_generator_agent = create_view_generator()
