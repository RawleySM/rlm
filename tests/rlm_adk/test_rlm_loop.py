"""Tests for RLM iteration loop (LoopAgent) and completion workflow (SequentialAgent).

This module tests the assembly and configuration of the RLM iteration loop
and completion workflow as specified in specs/phase4-test-specification.md.
"""

import pytest
from google.adk.agents import LoopAgent, SequentialAgent, LlmAgent, BaseAgent


class TestRLMIterationLoopStructure:
    """Tests for RLM iteration loop factory and configuration."""

    def test_make_rlm_iteration_loop_returns_loop_agent(self):
        """Factory returns LoopAgent."""
        from rlm_adk.agents.rlm_loop import make_rlm_iteration_loop

        result = make_rlm_iteration_loop()
        assert isinstance(result, LoopAgent)

    def test_loop_agent_has_correct_name(self):
        """Name is `rlm_iteration_loop`."""
        from rlm_adk.agents.rlm_loop import make_rlm_iteration_loop

        loop = make_rlm_iteration_loop()
        assert loop.name == "rlm_iteration_loop"

    def test_loop_agent_has_three_sub_agents(self):
        """Has code_generator, code_executor, completion_checker."""
        from rlm_adk.agents.rlm_loop import make_rlm_iteration_loop

        loop = make_rlm_iteration_loop()
        assert len(loop.sub_agents) == 3

    def test_loop_agent_max_iterations_default(self):
        """Default max_iterations is 10."""
        from rlm_adk.agents.rlm_loop import make_rlm_iteration_loop

        loop = make_rlm_iteration_loop()
        assert loop.max_iterations == 10

    def test_loop_agent_max_iterations_custom(self):
        """Custom max_iterations respected."""
        from rlm_adk.agents.rlm_loop import make_rlm_iteration_loop

        loop = make_rlm_iteration_loop(max_iterations=5)
        assert loop.max_iterations == 5

    def test_loop_agent_has_before_callback(self):
        """before_agent_callback attached."""
        from rlm_adk.agents.rlm_loop import make_rlm_iteration_loop

        loop = make_rlm_iteration_loop()
        assert loop.before_agent_callback is not None

    def test_loop_agent_has_after_callback(self):
        """after_agent_callback attached."""
        from rlm_adk.agents.rlm_loop import make_rlm_iteration_loop

        loop = make_rlm_iteration_loop()
        assert loop.after_agent_callback is not None


class TestRLMCompletionWorkflowStructure:
    """Tests for the full SequentialAgent workflow."""

    def test_make_rlm_completion_workflow_returns_sequential_agent(self):
        """Factory returns SequentialAgent."""
        from rlm_adk.agents.rlm_loop import make_rlm_completion_workflow

        result = make_rlm_completion_workflow()
        assert isinstance(result, SequentialAgent)

    def test_workflow_has_correct_name(self):
        """Name is `rlm_completion_workflow`."""
        from rlm_adk.agents.rlm_loop import make_rlm_completion_workflow

        workflow = make_rlm_completion_workflow()
        assert workflow.name == "rlm_completion_workflow"

    def test_workflow_has_three_stages(self):
        """Has context_loader, loop, formatter."""
        from rlm_adk.agents.rlm_loop import make_rlm_completion_workflow

        workflow = make_rlm_completion_workflow()
        assert len(workflow.sub_agents) == 3

    def test_workflow_first_stage_is_context_loader(self):
        """First stage loads context."""
        from rlm_adk.agents.rlm_loop import make_rlm_completion_workflow

        workflow = make_rlm_completion_workflow()
        assert workflow.sub_agents[0].name == "rlm_context_loader"

    def test_workflow_second_stage_is_iteration_loop(self):
        """Second stage is LoopAgent."""
        from rlm_adk.agents.rlm_loop import make_rlm_completion_workflow

        workflow = make_rlm_completion_workflow()
        assert workflow.sub_agents[1].name == "rlm_iteration_loop"

    def test_workflow_third_stage_is_result_formatter(self):
        """Third stage formats results."""
        from rlm_adk.agents.rlm_loop import make_rlm_completion_workflow

        workflow = make_rlm_completion_workflow()
        assert workflow.sub_agents[2].name == "rlm_result_formatter"


class TestSubAgentFactories:
    """Test individual agent factory functions."""

    # Code Generator Tests

    def test_make_code_generator_returns_llm_agent(self):
        """Factory returns LlmAgent."""
        from rlm_adk.agents.code_generator import make_code_generator

        result = make_code_generator()
        assert isinstance(result, LlmAgent)

    def test_code_generator_has_output_key(self):
        """Output key is `generated_code`."""
        from rlm_adk.agents.code_generator import make_code_generator

        agent = make_code_generator()
        assert agent.output_key == "generated_code"

    def test_code_generator_instruction_contains_rlm_prompt(self):
        """Uses RLM system prompt."""
        from rlm_adk.agents.code_generator import make_code_generator

        agent = make_code_generator()
        # Instruction should include llm_query function documentation
        assert "llm_query" in agent.instruction

    def test_code_generator_has_callbacks(self):
        """All 4 callbacks attached."""
        from rlm_adk.agents.code_generator import make_code_generator

        agent = make_code_generator()
        # Check all 4 callback types
        assert agent.before_agent_callback is not None
        assert agent.before_model_callback is not None
        assert agent.after_model_callback is not None
        assert agent.on_model_error_callback is not None

    # Code Executor Tests

    def test_make_code_executor_returns_llm_agent(self):
        """Factory returns LlmAgent."""
        from rlm_adk.agents.code_executor import make_code_executor

        result = make_code_executor()
        assert isinstance(result, LlmAgent)

    def test_code_executor_has_execute_tool(self):
        """Has execute_rlm_iteration tool."""
        from rlm_adk.agents.code_executor import make_code_executor

        agent = make_code_executor()
        assert agent.tools is not None
        assert len(agent.tools) > 0
        # Check that execute_rlm_iteration is in tools
        tool_names = [t.name if hasattr(t, 'name') else str(t) for t in agent.tools]
        assert "execute_rlm_iteration" in tool_names

    def test_code_executor_has_output_key(self):
        """Output key is `execution_result`."""
        from rlm_adk.agents.code_executor import make_code_executor

        agent = make_code_executor()
        assert agent.output_key == "execution_result"

    def test_code_executor_has_tool_callbacks(self):
        """Has before_tool and after_tool callbacks."""
        from rlm_adk.agents.code_executor import make_code_executor

        agent = make_code_executor()
        assert agent.before_tool_callback is not None
        assert agent.after_tool_callback is not None

    # Completion Checker Tests

    def test_completion_checker_is_base_agent(self):
        """RLMCompletionChecker extends BaseAgent."""
        from rlm_adk.agents.completion_checker import RLMCompletionChecker

        result = RLMCompletionChecker()
        assert isinstance(result, BaseAgent)

    def test_completion_checker_has_correct_name(self):
        """Default name is `rlm_completion_checker`."""
        from rlm_adk.agents.completion_checker import RLMCompletionChecker

        checker = RLMCompletionChecker()
        assert checker.name == "rlm_completion_checker"

    # Context Setup Tests

    def test_make_context_setup_agent_returns_llm_agent(self):
        """Factory returns LlmAgent."""
        from rlm_adk.agents.context_setup import make_context_setup_agent

        result = make_context_setup_agent()
        assert isinstance(result, LlmAgent)

    def test_context_setup_has_load_tool(self):
        """Has rlm_load_context tool."""
        from rlm_adk.agents.context_setup import make_context_setup_agent

        agent = make_context_setup_agent()
        assert agent.tools is not None
        assert len(agent.tools) > 0
        # Check that rlm_load_context is in tools
        tool_names = [t.name if hasattr(t, "name") else str(t) for t in agent.tools]
        assert "rlm_load_context" in tool_names

    # Result Formatter Tests

    def test_make_result_formatter_returns_llm_agent(self):
        """Factory returns LlmAgent."""
        from rlm_adk.agents.result_formatter import make_result_formatter

        result = make_result_formatter()
        assert isinstance(result, LlmAgent)

    def test_result_formatter_has_output_key(self):
        """Output key is `rlm_formatted_result`."""
        from rlm_adk.agents.result_formatter import make_result_formatter

        agent = make_result_formatter()
        assert agent.output_key == "rlm_formatted_result"
