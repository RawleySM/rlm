"""Import tests for RLM-ADK package.

Verifies that all modules and components can be imported successfully.
"""

import pytest


class TestPackageImports:
    """Test package-level imports."""

    def test_import_rlm_adk_package(self):
        """Test importing the main package."""
        import rlm_adk

        assert rlm_adk is not None

    def test_import_root_agent_from_package(self):
        """Test importing root_agent from package."""
        from rlm_adk import root_agent

        assert root_agent is not None


class TestToolsImports:
    """Test tools module imports."""

    def test_import_tools_module(self):
        """Test importing the tools module."""
        from rlm_adk import tools

        assert tools is not None

    def test_import_databricks_repl_tools(self):
        """Test importing Databricks REPL tools."""
        from rlm_adk.tools.databricks_repl import (
            execute_python_code,
            execute_sql_query,
            get_repl_session_state,
        )

        assert callable(execute_python_code)
        assert callable(execute_sql_query)
        assert callable(get_repl_session_state)

    def test_import_unity_catalog_tools(self):
        """Test importing Unity Catalog tools."""
        from rlm_adk.tools.unity_catalog import (
            create_view,
            get_volume_metadata,
            list_catalogs,
            list_schemas,
            list_tables,
            list_volumes,
            read_table_sample,
        )

        assert callable(list_catalogs)
        assert callable(list_schemas)
        assert callable(list_tables)
        assert callable(list_volumes)
        assert callable(get_volume_metadata)
        assert callable(read_table_sample)
        assert callable(create_view)

    def test_import_vendor_resolution_tools(self):
        """Test importing vendor resolution tools."""
        from rlm_adk.tools.vendor_resolution import (
            create_vendor_mapping,
            find_similar_vendors,
            get_masterdata_vendor,
            resolve_vendor_to_masterdata,
            search_vendor_by_attributes,
        )

        assert callable(find_similar_vendors)
        assert callable(resolve_vendor_to_masterdata)
        assert callable(get_masterdata_vendor)
        assert callable(create_vendor_mapping)
        assert callable(search_vendor_by_attributes)


class TestAgentsImports:
    """Test agents module imports."""

    def test_import_agents_module(self):
        """Test importing the agents module."""
        from rlm_adk import agents

        assert agents is not None

    def test_import_erp_analyzers(self):
        """Test importing ERP analyzer agents."""
        from rlm_adk.agents.erp_analyzer import (
            alpha_erp_analyzer,
            beta_erp_analyzer,
            gamma_erp_analyzer,
            make_erp_analyzer,
        )

        assert alpha_erp_analyzer is not None
        assert beta_erp_analyzer is not None
        assert gamma_erp_analyzer is not None
        assert callable(make_erp_analyzer)

    def test_import_vendor_matcher(self):
        """Test importing vendor matcher agent."""
        from rlm_adk.agents.vendor_matcher import vendor_matcher_agent

        assert vendor_matcher_agent is not None

    def test_import_view_generator(self):
        """Test importing view generator agent."""
        from rlm_adk.agents.view_generator import view_generator_agent

        assert view_generator_agent is not None


class TestTestingUtils:
    """Test the testing utilities."""

    def test_create_tool_context(self):
        """Test create_tool_context factory."""
        from rlm_adk.testing import create_tool_context

        ctx = create_tool_context({"test": 123})
        assert ctx.state["test"] == 123
