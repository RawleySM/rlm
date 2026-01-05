"""Import tests for RLM-ADK package.

Verifies that all modules and components can be imported successfully.
Tests are split between those that work without google-adk and those that require it.
"""

import pytest

from rlm_adk._compat import ADK_AVAILABLE

# Skip marker for tests requiring ADK
requires_adk = pytest.mark.skipif(
    not ADK_AVAILABLE,
    reason="google-adk not installed"
)


class TestPackageImports:
    """Test package-level imports (work without ADK)."""

    def test_import_rlm_adk_package(self):
        """Test importing the main package."""
        import rlm_adk

        assert rlm_adk is not None

    def test_import_root_agent_from_package(self):
        """Test importing root_agent from package (lazy proxy)."""
        from rlm_adk import root_agent

        # Should be a lazy proxy, not None
        assert root_agent is not None

    def test_import_adk_available_flag(self):
        """Test ADK_AVAILABLE flag is importable."""
        from rlm_adk import ADK_AVAILABLE

        assert isinstance(ADK_AVAILABLE, bool)


class TestToolsImports:
    """Test tools module imports (all work without ADK)."""

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

    def test_import_all_tools_from_tools_init(self):
        """Test importing all tools from tools __init__."""
        from rlm_adk.tools import (
            # Databricks REPL
            execute_python_code,
            execute_sql_query,
            get_repl_session_state,
            # Unity Catalog
            create_view,
            get_volume_metadata,
            list_catalogs,
            list_schemas,
            list_tables,
            list_volumes,
            read_table_sample,
            # Vendor Resolution
            create_vendor_mapping,
            find_similar_vendors,
            get_masterdata_vendor,
            resolve_vendor_to_masterdata,
            search_vendor_by_attributes,
        )

        # All should be callable
        all_tools = [
            execute_python_code,
            execute_sql_query,
            get_repl_session_state,
            list_catalogs,
            list_schemas,
            list_tables,
            list_volumes,
            get_volume_metadata,
            read_table_sample,
            create_view,
            find_similar_vendors,
            resolve_vendor_to_masterdata,
            get_masterdata_vendor,
            create_vendor_mapping,
            search_vendor_by_attributes,
        ]

        for tool in all_tools:
            assert callable(tool)


class TestAgentsImports:
    """Test agents module imports (lazy proxies work without ADK)."""

    def test_import_agents_module(self):
        """Test importing the agents module."""
        from rlm_adk import agents

        assert agents is not None

    def test_import_erp_analyzers(self):
        """Test importing ERP analyzer agents (lazy proxies)."""
        from rlm_adk.agents.erp_analyzer import (
            alpha_erp_analyzer,
            beta_erp_analyzer,
            gamma_erp_analyzer,
            make_erp_analyzer,
        )

        # Lazy proxies should exist
        assert alpha_erp_analyzer is not None
        assert beta_erp_analyzer is not None
        assert gamma_erp_analyzer is not None
        assert callable(make_erp_analyzer)

    def test_import_vendor_matcher(self):
        """Test importing vendor matcher agent (lazy proxy)."""
        from rlm_adk.agents.vendor_matcher import vendor_matcher_agent

        assert vendor_matcher_agent is not None

    def test_import_view_generator(self):
        """Test importing view generator agent (lazy proxy)."""
        from rlm_adk.agents.view_generator import view_generator_agent

        assert view_generator_agent is not None

    def test_import_all_agents_from_agents_init(self):
        """Test importing all agents from agents __init__."""
        from rlm_adk.agents import (
            alpha_erp_analyzer,
            beta_erp_analyzer,
            gamma_erp_analyzer,
            make_erp_analyzer,
            vendor_matcher_agent,
            view_generator_agent,
        )

        assert alpha_erp_analyzer is not None
        assert beta_erp_analyzer is not None
        assert gamma_erp_analyzer is not None
        assert callable(make_erp_analyzer)
        assert vendor_matcher_agent is not None
        assert view_generator_agent is not None


class TestAgentModuleImports:
    """Test main agent module imports (lazy proxies work without ADK)."""

    def test_import_agent_module(self):
        """Test importing the agent module."""
        from rlm_adk import agent

        assert agent is not None

    def test_import_workflow_agents(self):
        """Test importing workflow agents (lazy proxies)."""
        from rlm_adk.agent import (
            parallel_erp_analysis,
            vendor_resolution_pipeline,
        )

        assert parallel_erp_analysis is not None
        assert vendor_resolution_pipeline is not None


class TestCompatModule:
    """Test the compatibility module."""

    def test_import_compat(self):
        """Test importing compat module."""
        from rlm_adk._compat import (
            ADK_AVAILABLE,
            SimpleToolContext,
            ToolContextProtocol,
            check_adk_available,
            create_tool_context,
        )

        assert isinstance(ADK_AVAILABLE, bool)
        assert SimpleToolContext is not None
        assert ToolContextProtocol is not None
        assert callable(check_adk_available)
        assert callable(create_tool_context)

    def test_simple_tool_context(self):
        """Test SimpleToolContext works correctly."""
        from rlm_adk._compat import SimpleToolContext

        ctx = SimpleToolContext({"key": "value"})
        assert ctx.state["key"] == "value"

        ctx.state["new_key"] = "new_value"
        assert ctx.state["new_key"] == "new_value"

    def test_create_tool_context(self):
        """Test create_tool_context factory."""
        from rlm_adk._compat import create_tool_context

        ctx = create_tool_context({"test": 123})
        assert ctx.state["test"] == 123
