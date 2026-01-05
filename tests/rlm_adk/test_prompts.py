#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=8.0.0",
# ]
# ///
"""Test suite for rlm_adk/prompts.py - System prompt composition.

Tests cover:
1. TestSystemPromptComposition - Base system prompt composition (8 tests)
2. TestCodeGeneratorInstruction - Dynamic code generator instruction building (9 tests)
3. TestRootAgentInstruction - Root agent instruction constant (5 tests)
4. TestHealthcareExtension - Healthcare vendor extension content (5 tests)

Test execution:
    uv run /home/rawleysm/dev/rlm/tests/rlm_adk/test_prompts.py
"""

import pytest

from rlm_adk.prompts import (
    HEALTHCARE_VENDOR_EXTENSION,
    ROOT_AGENT_INSTRUCTION,
    get_code_generator_instruction,
    get_rlm_system_prompt,
)


# =============================================================================
# Test Class 1: TestSystemPromptComposition
# =============================================================================


class TestSystemPromptComposition:
    """Test the base system prompt composition.

    Verifies get_rlm_system_prompt() correctly includes base RLM content
    and optionally appends healthcare extension.
    """

    def test_get_rlm_system_prompt_includes_base(self):
        """Contains RLM_SYSTEM_PROMPT content."""
        prompt = get_rlm_system_prompt(include_healthcare_extension=False)

        # Base prompt must document key RLM patterns
        assert "llm_query" in prompt, "Should document llm_query function"
        assert "context" in prompt, "Should document context variable"

    def test_get_rlm_system_prompt_with_healthcare_extension(self):
        """Contains healthcare content when enabled."""
        prompt = get_rlm_system_prompt(include_healthcare_extension=True)

        # Should include healthcare-specific content
        assert "Healthcare" in prompt, "Should mention Healthcare"
        assert "vendor" in prompt.lower(), "Should mention vendor management"

    def test_get_rlm_system_prompt_without_healthcare_extension(self):
        """Excludes healthcare when disabled."""
        prompt = get_rlm_system_prompt(include_healthcare_extension=False)

        # Should NOT include healthcare content when disabled
        assert "Healthcare" not in prompt, "Should not mention Healthcare when disabled"

    def test_rlm_system_prompt_mentions_llm_query(self):
        """Documents llm_query function."""
        prompt = get_rlm_system_prompt(include_healthcare_extension=False)

        # Should document the llm_query function
        assert "llm_query(" in prompt, "Should show llm_query function signature"

    def test_rlm_system_prompt_mentions_llm_query_batched(self):
        """Documents llm_query_batched."""
        prompt = get_rlm_system_prompt(include_healthcare_extension=False)

        # Should document batched queries
        assert "llm_query_batched" in prompt, "Should document llm_query_batched"

    def test_rlm_system_prompt_mentions_final(self):
        """Documents FINAL pattern."""
        prompt = get_rlm_system_prompt(include_healthcare_extension=False)

        # Should document FINAL() termination pattern
        assert "FINAL(" in prompt, "Should show FINAL() pattern"

    def test_rlm_system_prompt_mentions_final_var(self):
        """Documents FINAL_VAR pattern."""
        prompt = get_rlm_system_prompt(include_healthcare_extension=False)

        # Should document FINAL_VAR() termination pattern
        assert "FINAL_VAR(" in prompt, "Should show FINAL_VAR() pattern"

    def test_rlm_system_prompt_mentions_context(self):
        """Documents context variable."""
        prompt = get_rlm_system_prompt(include_healthcare_extension=False)

        # Should document the context variable
        assert "context" in prompt, "Should document context variable"


# =============================================================================
# Test Class 2: TestCodeGeneratorInstruction
# =============================================================================


class TestCodeGeneratorInstruction:
    """Test dynamic code generator instruction building.

    Verifies get_code_generator_instruction() correctly composes
    instruction with dynamic state values and placeholders.
    """

    def test_get_code_generator_instruction_returns_string(self):
        """Returns non-empty string."""
        result = get_code_generator_instruction()

        assert isinstance(result, str), "Should return a string"
        assert len(result) > 0, "Should return non-empty string"

    def test_code_generator_instruction_includes_base_prompt(self):
        """Contains base RLM prompt."""
        instruction = get_code_generator_instruction()

        # Should include base RLM system prompt
        assert "llm_query" in instruction, "Should include RLM system prompt"

    def test_code_generator_instruction_includes_healthcare(self):
        """Contains healthcare extension."""
        instruction = get_code_generator_instruction()

        # Should include healthcare extension (default behavior)
        assert "Healthcare" in instruction, "Should include healthcare extension"

    def test_code_generator_instruction_includes_context_placeholder(self):
        """Has context_description placeholder."""
        # Call without context_description
        instruction = get_code_generator_instruction()

        # Should have placeholder or actual description
        assert (
            "Context Description" in instruction or
            "context" in instruction.lower()
        ), "Should have context section"

    def test_code_generator_instruction_includes_history_placeholder(self):
        """Has iteration_history placeholder."""
        # Call without iteration_history (uses default)
        instruction = get_code_generator_instruction()

        # Should have placeholder or actual history
        assert (
            "Previous Iterations" in instruction or
            "No previous iterations" in instruction
        ), "Should have iteration history section"

    def test_code_generator_instruction_includes_query_placeholder(self):
        """Has user_query placeholder."""
        # Call without user_query
        instruction = get_code_generator_instruction()

        # Should have placeholder or actual query
        assert (
            "User Query" in instruction or
            "Awaiting user query" in instruction or
            "query" in instruction.lower()
        ), "Should have user query section"

    def test_code_generator_instruction_with_actual_context(self):
        """Includes actual context when provided."""
        actual_context = "Vendor data from Alpha hospital chain (5000 records)"
        instruction = get_code_generator_instruction(context_description=actual_context)

        # Should include the actual description
        assert actual_context in instruction, "Should include actual context description"

    def test_code_generator_instruction_with_actual_history(self):
        """Includes actual history when provided."""
        actual_history = "Iteration 1: Loaded vendor data\nIteration 2: Found 42 duplicates"
        instruction = get_code_generator_instruction(iteration_history=actual_history)

        # Should include the actual history
        assert actual_history in instruction, "Should include actual iteration history"

    def test_code_generator_instruction_empty_values_use_defaults(self):
        """Empty strings use default placeholders."""
        # Pass empty strings explicitly
        instruction = get_code_generator_instruction(
            context_description="",
            iteration_history="",
            user_query=""
        )

        # Should use default placeholders for empty values
        assert (
            "No context loaded yet" in instruction or
            "No previous iterations" in instruction or
            "Awaiting user query" in instruction
        ), "Should use default placeholders for empty values"


# =============================================================================
# Test Class 3: TestRootAgentInstruction
# =============================================================================


class TestRootAgentInstruction:
    """Test the root agent instruction constant.

    Verifies ROOT_AGENT_INSTRUCTION includes all required components
    for the healthcare data scientist agent.
    """

    def test_root_agent_instruction_exists(self):
        """ROOT_AGENT_INSTRUCTION is defined."""
        assert ROOT_AGENT_INSTRUCTION is not None, "Should be defined"
        assert isinstance(ROOT_AGENT_INSTRUCTION, str), "Should be a string"
        assert len(ROOT_AGENT_INSTRUCTION) > 0, "Should be non-empty"

    def test_root_agent_instruction_includes_rlm_prompt(self):
        """Contains RLM system prompt."""
        # Should include base RLM system prompt content
        assert "llm_query" in ROOT_AGENT_INSTRUCTION, "Should include RLM system prompt"

    def test_root_agent_instruction_includes_healthcare(self):
        """Contains healthcare extension."""
        # Should include healthcare extension
        assert "Healthcare" in ROOT_AGENT_INSTRUCTION, "Should include healthcare extension"

    def test_root_agent_instruction_documents_workflows(self):
        """Documents available workflows."""
        # Should document the RLM completion workflow
        assert (
            "rlm_completion_workflow" in ROOT_AGENT_INSTRUCTION
        ), "Should document rlm_completion_workflow"

    def test_root_agent_instruction_documents_tools(self):
        """Documents direct RLM tools."""
        # Should document available RLM tools
        assert "rlm_load_context" in ROOT_AGENT_INSTRUCTION, "Should document rlm_load_context"


# =============================================================================
# Test Class 4: TestHealthcareExtension
# =============================================================================


class TestHealthcareExtension:
    """Test the healthcare vendor extension content.

    Verifies HEALTHCARE_VENDOR_EXTENSION includes all required
    domain-specific content.
    """

    def test_healthcare_extension_mentions_hospital_chains(self):
        """Mentions Alpha, Beta, Gamma."""
        # Should mention all three hospital chains
        assert "Alpha" in HEALTHCARE_VENDOR_EXTENSION, "Should mention Alpha chain"
        assert "Beta" in HEALTHCARE_VENDOR_EXTENSION, "Should mention Beta chain"
        assert "Gamma" in HEALTHCARE_VENDOR_EXTENSION, "Should mention Gamma chain"

    def test_healthcare_extension_mentions_masterdata(self):
        """Mentions masterdata."""
        # Should mention masterdata
        assert "masterdata" in HEALTHCARE_VENDOR_EXTENSION.lower(), (
            "Should mention masterdata"
        )

    def test_healthcare_extension_mentions_tax_id(self):
        """Mentions Tax ID for matching."""
        # Should mention Tax ID for vendor matching
        assert "tax id" in HEALTHCARE_VENDOR_EXTENSION.lower(), (
            "Should mention Tax ID"
        )

    def test_healthcare_extension_mentions_duns(self):
        """Mentions DUNS number."""
        # Should mention DUNS number
        assert "duns" in HEALTHCARE_VENDOR_EXTENSION.lower(), (
            "Should mention DUNS number"
        )

    def test_healthcare_extension_mentions_chunking(self):
        """Recommends chunking large datasets."""
        # Should recommend chunking for large datasets
        assert "chunk" in HEALTHCARE_VENDOR_EXTENSION.lower(), (
            "Should recommend chunking"
        )


# =============================================================================
# Test Execution
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
