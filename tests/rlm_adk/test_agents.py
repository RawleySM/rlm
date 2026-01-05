"""Tests for RLM-ADK agents.

Tests the agent definitions and structure. Many tests require
google-adk to be installed and will be skipped if not available.
"""

import pytest

from rlm_adk._compat import ADK_AVAILABLE

# Skip marker for tests requiring ADK
requires_adk = pytest.mark.skipif(
    not ADK_AVAILABLE,
    reason="google-adk not installed"
)


class TestAgentImportsWithoutADK:
    """Test that agent modules can be imported without ADK."""

    def test_import_package(self):
        """Test importing the main package."""
        import rlm_adk

        assert rlm_adk is not None

    def test_adk_available_flag(self):
        """Test ADK_AVAILABLE flag is accessible."""
        from rlm_adk import ADK_AVAILABLE

        # Should be a boolean
        assert isinstance(ADK_AVAILABLE, bool)

    def test_lazy_agents_exist(self):
        """Test lazy agent proxies are accessible."""
        from rlm_adk import (
            parallel_erp_analysis,
            root_agent,
            vendor_resolution_pipeline,
        )

        # These should be lazy proxies, not None
        assert root_agent is not None
        assert parallel_erp_analysis is not None
        assert vendor_resolution_pipeline is not None


@requires_adk
class TestAgentImportsWithADK:
    """Tests that require google-adk to be installed."""

    def test_import_root_agent_properties(self):
        """Test accessing root agent properties."""
        from rlm_adk import root_agent

        assert root_agent.name == "rlm_data_scientist"

    def test_import_from_agent_module(self):
        """Test importing from the agent module."""
        from rlm_adk.agent import (
            parallel_erp_analysis,
            root_agent,
        )

        assert root_agent.name == "rlm_data_scientist"
        assert parallel_erp_analysis.name == "parallel_erp_analysis"


class TestAgentStructureWithoutADK:
    """Test agent structure that works without ADK."""

    def test_erp_analyzer_lazy_proxy_properties(self):
        """Test ERP analyzer lazy proxy has expected properties."""
        from rlm_adk.agents import alpha_erp_analyzer

        # These properties work without loading the actual agent
        assert alpha_erp_analyzer.name == "hospital_chain_alpha_erp_analyzer"
        assert alpha_erp_analyzer.output_key == "hospital_chain_alpha_vendor_analysis"

    def test_vendor_matcher_lazy_proxy_properties(self):
        """Test vendor matcher lazy proxy has expected properties."""
        from rlm_adk.agents import vendor_matcher_agent

        assert vendor_matcher_agent.name == "vendor_matcher"
        assert vendor_matcher_agent.output_key == "vendor_resolution_results"

    def test_view_generator_lazy_proxy_properties(self):
        """Test view generator lazy proxy has expected properties."""
        from rlm_adk.agents import view_generator_agent

        assert view_generator_agent.name == "view_generator"
        assert view_generator_agent.output_key == "created_views"


@requires_adk
class TestAgentStructureWithADK:
    """Test agent structure that requires ADK."""

    def test_root_agent_has_tools(self):
        """Test that root agent has required tools."""
        from rlm_adk import root_agent

        assert root_agent.tools is not None
        assert len(root_agent.tools) > 0

    def test_root_agent_has_sub_agents(self):
        """Test that root agent has sub-agents for delegation."""
        from rlm_adk import root_agent

        assert root_agent.sub_agents is not None
        assert len(root_agent.sub_agents) > 0

    def test_parallel_erp_analysis_structure(self):
        """Test parallel ERP analysis agent structure."""
        from rlm_adk.agent import parallel_erp_analysis

        # Should have sub-agents for each hospital chain
        assert parallel_erp_analysis.sub_agents is not None
        assert len(parallel_erp_analysis.sub_agents) == 3  # Alpha, Beta, Gamma

    def test_vendor_resolution_pipeline_structure(self):
        """Test vendor resolution pipeline structure."""
        from rlm_adk.agent import vendor_resolution_pipeline

        # Should be a sequential pipeline with stages
        assert vendor_resolution_pipeline.sub_agents is not None
        assert len(vendor_resolution_pipeline.sub_agents) >= 3

    def test_erp_analyzer_factory(self):
        """Test the ERP analyzer factory function."""
        from rlm_adk.agents.erp_analyzer import make_erp_analyzer

        custom_analyzer = make_erp_analyzer(
            hospital_chain="Test Hospital",
            catalog_name="test_catalog",
        )

        assert custom_analyzer.name == "test_catalog_erp_analyzer"
        assert "Test Hospital" in custom_analyzer.instruction
        assert "test_catalog" in custom_analyzer.instruction

    def test_vendor_matcher_has_resolution_tools(self):
        """Test vendor matcher has vendor resolution tools."""
        from rlm_adk.agents.vendor_matcher import vendor_matcher_agent

        tool_names = [
            t.__name__ if callable(t) else str(t) for t in vendor_matcher_agent.tools
        ]

        # Should have vendor resolution tools
        assert any("similar" in name.lower() for name in tool_names)
        assert any("resolve" in name.lower() or "mapping" in name.lower() for name in tool_names)

    def test_view_generator_has_create_view_tool(self):
        """Test view generator has view creation tools."""
        from rlm_adk.agents.view_generator import view_generator_agent

        tool_names = [
            t.__name__ if callable(t) else str(t) for t in view_generator_agent.tools
        ]

        assert any("view" in name.lower() for name in tool_names)


@requires_adk
class TestAgentOutputKeys:
    """Test that agents have appropriate output keys configured."""

    def test_erp_analyzers_have_output_keys(self):
        """Test ERP analyzers have output keys for state storage."""
        from rlm_adk.agents import (
            alpha_erp_analyzer,
            beta_erp_analyzer,
            gamma_erp_analyzer,
        )

        assert alpha_erp_analyzer.output_key == "hospital_chain_alpha_vendor_analysis"
        assert beta_erp_analyzer.output_key == "hospital_chain_beta_vendor_analysis"
        assert gamma_erp_analyzer.output_key == "hospital_chain_gamma_vendor_analysis"

    def test_vendor_matcher_has_output_key(self):
        """Test vendor matcher has output key."""
        from rlm_adk.agents.vendor_matcher import vendor_matcher_agent

        assert vendor_matcher_agent.output_key == "vendor_resolution_results"

    def test_view_generator_has_output_key(self):
        """Test view generator has output key."""
        from rlm_adk.agents.view_generator import view_generator_agent

        assert view_generator_agent.output_key == "created_views"
