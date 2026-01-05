"""Vendor Matcher agent for masterdata resolution.

Provides an agent specialized in matching vendor records across
multiple hospital chains and resolving them to masterdata entities.

Requires: google-adk (pip install google-adk)
"""

from rlm_adk._compat import ADK_AVAILABLE, check_adk_available

_vendor_matcher_agent = None


def _create_vendor_matcher():
    """Create the vendor matcher agent."""
    check_adk_available()

    from google.adk.agents import Agent

    from rlm_adk.tools.databricks_repl import execute_python_code, execute_sql_query
    from rlm_adk.tools.vendor_resolution import (
        create_vendor_mapping,
        find_similar_vendors,
        get_masterdata_vendor,
        resolve_vendor_to_masterdata,
        search_vendor_by_attributes,
    )

    return Agent(
        name="vendor_matcher",
        model="gemini-2.0-flash",
        description="Matches and resolves vendor records from hospital ERPs to masterdata golden records.",
        instruction="""You are a master data management specialist focused on vendor entity resolution.

**Your Mission:**
Resolve vendor instances from multiple hospital chain ERP databases to unified
masterdata vendor entities. Create accurate mappings that consolidate the same
real-world vendor across different hospital systems.

**Available Context (from parallel ERP analysis):**
- {hospital_chain_alpha_vendor_analysis}: Vendor analysis from Alpha Hospital Chain
- {hospital_chain_beta_vendor_analysis}: Vendor analysis from Beta Hospital Network
- {hospital_chain_gamma_vendor_analysis}: Vendor analysis from Gamma Health System

**Resolution Strategy:**

1. **Exact Match First (Highest Confidence: 0.95-1.0):**
   - Match by Tax ID (EIN) - this is the most reliable identifier
   - Match by DUNS number if available
   - Use `search_vendor_by_attributes` for exact attribute matching

2. **Fuzzy Name Matching (Medium Confidence: 0.70-0.90):**
   - Use `find_similar_vendors` to identify potential name matches
   - Consider common variations: Corp/Corporation, Inc/Incorporated, LLC
   - Watch for abbreviations and typos

3. **Address Correlation (Supporting Evidence):**
   - Same or similar addresses increase match confidence
   - Different addresses don't necessarily mean different vendors

**Decision Rules:**
- If Tax ID matches exactly across vendors -> Same vendor (confidence 0.98)
- If name similarity > 0.85 AND same Tax ID -> Same vendor (confidence 0.95)
- If name similarity > 0.85 AND address similar -> Likely same vendor (confidence 0.80)
- If name similarity > 0.75 AND no conflicting info -> Possible match (confidence 0.70)
- Below 0.70 confidence -> Flag for human review

**Actions:**

1. For matched vendors WITH existing masterdata:
   - Use `resolve_vendor_to_masterdata` to create mappings

2. For matched vendors WITHOUT existing masterdata:
   - Use `create_vendor_mapping` to create a new golden record

3. For uncertain matches:
   - Document the uncertainty in your output
   - Recommend manual review

**Output Format:**
Provide a resolution report including:
- Total vendors analyzed from each hospital
- Exact matches found
- Fuzzy matches created
- New masterdata records created
- Flagged for review
- Summary statistics

Store your resolution results in state key 'vendor_resolution_results'.
""",
        tools=[
            find_similar_vendors,
            search_vendor_by_attributes,
            get_masterdata_vendor,
            resolve_vendor_to_masterdata,
            create_vendor_mapping,
            execute_sql_query,
            execute_python_code,
        ],
        output_key="vendor_resolution_results",
    )


class _LazyVendorMatcher:
    """Lazy proxy for vendor matcher agent."""

    def __init__(self):
        self._agent = None

    def _ensure_loaded(self):
        if self._agent is None:
            self._agent = _create_vendor_matcher()
        return self._agent

    @property
    def name(self):
        return "vendor_matcher"

    @property
    def output_key(self):
        return "vendor_resolution_results"

    @property
    def tools(self):
        return self._ensure_loaded().tools

    def __getattr__(self, name):
        return getattr(self._ensure_loaded(), name)

    def __repr__(self):
        if ADK_AVAILABLE:
            return repr(self._ensure_loaded())
        return "<LazyVendorMatcher - google-adk not installed>"


vendor_matcher_agent = _LazyVendorMatcher()
