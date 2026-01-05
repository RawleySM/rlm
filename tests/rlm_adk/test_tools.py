"""Tests for RLM-ADK tools.

Tests the Databricks REPL, Unity Catalog, and vendor resolution tools
using simulated responses (no actual Databricks connection required).
These tests do NOT require google-adk to be installed.
"""

import os

import pytest

from rlm_adk._compat import SimpleToolContext


@pytest.fixture(autouse=True)
def clear_databricks_env(monkeypatch):
    """Clear Databricks environment variables to force simulation mode."""
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.delenv("DATABRICKS_CLUSTER_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_SQL_WAREHOUSE_ID", raising=False)


class TestDatabricksREPLTools:
    """Tests for Databricks REPL tools."""

    def test_execute_python_code_simple(self):
        """Test executing simple Python code."""
        from rlm_adk.tools.databricks_repl import execute_python_code

        ctx = SimpleToolContext({"repl_session_id": "test_session"})
        result = execute_python_code("x = 1 + 1", ctx)

        assert result["status"] == "success"

    def test_execute_python_code_with_output(self):
        """Test Python code that produces output."""
        from rlm_adk.tools.databricks_repl import execute_python_code

        ctx = SimpleToolContext({"repl_session_id": "test_session_output"})
        result = execute_python_code("print('hello world')", ctx)

        assert result["status"] == "success"
        assert result.get("output") == "hello world\n"

    def test_execute_python_code_error_handling(self):
        """Test Python code error handling."""
        from rlm_adk.tools.databricks_repl import execute_python_code

        ctx = SimpleToolContext({"repl_session_id": "test_session_error"})
        result = execute_python_code("raise ValueError('test error')", ctx)

        assert result["status"] == "error"
        assert "test error" in result["error_message"]

    def test_execute_sql_query_show_catalogs(self):
        """Test SQL query for showing catalogs."""
        from rlm_adk.tools.databricks_repl import execute_sql_query

        ctx = SimpleToolContext()
        result = execute_sql_query("SHOW CATALOGS", ctx)

        assert result["status"] == "success"
        assert "catalogs" in str(result.get("columns", [])).lower() or result.get("data")

    def test_execute_sql_query_select(self):
        """Test SQL SELECT query."""
        from rlm_adk.tools.databricks_repl import execute_sql_query

        ctx = SimpleToolContext()
        result = execute_sql_query("SELECT * FROM vendors LIMIT 10", ctx)

        assert result["status"] == "success"
        assert "columns" in result or "data" in result

    def test_get_repl_session_state(self):
        """Test getting REPL session state."""
        from rlm_adk.tools.databricks_repl import get_repl_session_state

        ctx = SimpleToolContext({"repl_session_id": "test_state_session"})
        result = get_repl_session_state(ctx)

        assert result["status"] == "success"
        assert "variables" in result
        assert "session_id" in result


class TestUnityCatalogTools:
    """Tests for Unity Catalog tools."""

    def test_list_catalogs(self):
        """Test listing Unity Catalogs."""
        from rlm_adk.tools.unity_catalog import list_catalogs

        ctx = SimpleToolContext()
        result = list_catalogs(ctx)

        assert result["status"] == "success"
        assert "catalogs" in result
        assert result["count"] > 0

        # Verify expected catalogs are present
        catalog_names = [c["name"] for c in result["catalogs"]]
        assert "masterdata_vendors" in catalog_names

    def test_list_schemas(self):
        """Test listing schemas in a catalog."""
        from rlm_adk.tools.unity_catalog import list_schemas

        ctx = SimpleToolContext()
        result = list_schemas("hospital_chain_alpha", ctx)

        assert result["status"] == "success"
        assert result["catalog"] == "hospital_chain_alpha"
        assert "schemas" in result
        assert result["count"] > 0

    def test_list_tables(self):
        """Test listing tables in a schema."""
        from rlm_adk.tools.unity_catalog import list_tables

        ctx = SimpleToolContext()
        result = list_tables("hospital_chain_alpha", "erp_vendors", ctx)

        assert result["status"] == "success"
        assert result["catalog"] == "hospital_chain_alpha"
        assert result["schema"] == "erp_vendors"
        assert "tables" in result

    def test_list_volumes(self):
        """Test listing volumes in a schema."""
        from rlm_adk.tools.unity_catalog import list_volumes

        ctx = SimpleToolContext()
        result = list_volumes("hospital_chain_alpha", "erp_vendors", ctx)

        assert result["status"] == "success"
        assert "volumes" in result

    def test_get_volume_metadata(self):
        """Test getting volume metadata."""
        from rlm_adk.tools.unity_catalog import get_volume_metadata

        ctx = SimpleToolContext()
        result = get_volume_metadata(
            "hospital_chain_alpha", "erp_vendors", "raw_exports", ctx
        )

        assert result["status"] == "success"
        assert "volume" in result
        assert "full_name" in result["volume"]

    def test_read_table_sample(self):
        """Test reading a sample from a table."""
        from rlm_adk.tools.unity_catalog import read_table_sample

        ctx = SimpleToolContext()
        result = read_table_sample(
            "hospital_chain_alpha", "erp_vendors", "vendors", 10, ctx
        )

        assert result["status"] == "success"
        assert "columns" in result
        assert "data" in result
        assert result["row_count"] <= 10

    def test_create_view(self):
        """Test creating a view."""
        from rlm_adk.tools.unity_catalog import create_view

        ctx = SimpleToolContext()
        result = create_view(
            "healthcare_main",
            "analytics",
            "test_view",
            "SELECT * FROM some_table",
            True,
            ctx,
        )

        assert result["status"] == "success"
        assert "view" in result
        assert result["view"] == "healthcare_main.analytics.test_view"


class TestVendorResolutionTools:
    """Tests for vendor resolution tools."""

    def test_find_similar_vendors(self):
        """Test finding similar vendors."""
        from rlm_adk.tools.vendor_resolution import find_similar_vendors

        ctx = SimpleToolContext()
        result = find_similar_vendors("MedSupply", "all", 0.7, ctx)

        assert result["status"] == "success"
        assert "matches" in result
        assert "query" in result

        # All matches should meet threshold
        for match in result["matches"]:
            assert match["similarity_score"] >= 0.7

    def test_find_similar_vendors_with_hospital_filter(self):
        """Test finding similar vendors filtered by hospital chain."""
        from rlm_adk.tools.vendor_resolution import find_similar_vendors

        ctx = SimpleToolContext()
        result = find_similar_vendors(
            "MedSupply", "hospital_chain_alpha", 0.7, ctx
        )

        assert result["status"] == "success"

        # All matches should be from specified hospital
        for match in result["matches"]:
            assert match["hospital_chain"] == "hospital_chain_alpha"

    def test_resolve_vendor_to_masterdata(self):
        """Test resolving a vendor to masterdata."""
        from rlm_adk.tools.vendor_resolution import resolve_vendor_to_masterdata

        ctx = SimpleToolContext()
        result = resolve_vendor_to_masterdata(
            "ALPHA-V001", "hospital_chain_alpha", "MD-001", 0.95, ctx
        )

        assert result["status"] == "success"
        assert "mapping" in result
        assert result["mapping"]["source_vendor_id"] == "ALPHA-V001"
        assert result["mapping"]["masterdata_vendor_id"] == "MD-001"
        assert result["mapping"]["confidence_score"] == 0.95

    def test_get_masterdata_vendor(self):
        """Test getting masterdata vendor details."""
        from rlm_adk.tools.vendor_resolution import get_masterdata_vendor

        ctx = SimpleToolContext()
        result = get_masterdata_vendor("MD-001", ctx)

        assert result["status"] == "success"
        assert "vendor" in result
        assert result["vendor"]["masterdata_id"] == "MD-001"
        assert "canonical_name" in result["vendor"]
        assert "source_mappings" in result["vendor"]

    def test_get_masterdata_vendor_not_found(self):
        """Test getting non-existent masterdata vendor."""
        from rlm_adk.tools.vendor_resolution import get_masterdata_vendor

        ctx = SimpleToolContext()
        result = get_masterdata_vendor("MD-NONEXISTENT", ctx)

        assert result["status"] == "error"
        assert "not found" in result["error_message"].lower()

    def test_create_vendor_mapping(self):
        """Test creating a new masterdata vendor."""
        from rlm_adk.tools.vendor_resolution import create_vendor_mapping

        ctx = SimpleToolContext()
        result = create_vendor_mapping(
            canonical_name="New Vendor Corp",
            tax_id="99-9999999",
            duns_number="999999999",
            primary_address="999 New St, New York, NY",
            source_vendor_ids=[
                {"vendor_id": "ALPHA-V999", "hospital_chain": "hospital_chain_alpha"},
                {"vendor_id": "BETA-V999", "hospital_chain": "hospital_chain_beta"},
            ],
            tool_context=ctx,
        )

        assert result["status"] == "success"
        assert "vendor" in result
        assert result["vendor"]["canonical_name"] == "New Vendor Corp"
        assert result["mappings_created"] == 2

    def test_search_vendor_by_attributes(self):
        """Test searching vendors by attributes."""
        from rlm_adk.tools.vendor_resolution import search_vendor_by_attributes

        ctx = SimpleToolContext()
        result = search_vendor_by_attributes("12-3456789", "", "", ctx)

        assert result["status"] == "success"
        assert "hospital_vendors" in result
        assert "masterdata_vendors" in result

        # Vendors with matching tax ID should be found
        for vendor in result["hospital_vendors"]:
            assert vendor["tax_id"] == "12-3456789"
