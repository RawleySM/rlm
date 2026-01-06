"""RLM iteration loop using ADK LoopAgent.

Assembles the code_generator, code_executor, and completion_checker
into a LoopAgent workflow with callbacks for state management.
"""

from __future__ import annotations

from google.adk.agents import LoopAgent, SequentialAgent

from rlm_adk.agents.code_executor import make_code_executor
from rlm_adk.agents.code_generator import make_code_generator
from rlm_adk.agents.completion_checker import RLMCompletionChecker
from rlm_adk.agents.context_setup import make_context_setup_agent
from rlm_adk.agents.result_formatter import make_result_formatter
from rlm_adk.callbacks import get_rlm_loop_callbacks


def make_rlm_iteration_loop(max_iterations: int = 10) -> LoopAgent:
    """Create the RLM iteration loop with callbacks.

    Args:
        max_iterations: Maximum iterations before forced termination.

    Returns:
        LoopAgent configured for RLM iteration.
    """
    callbacks = get_rlm_loop_callbacks()

    return LoopAgent(
        name="rlm_iteration_loop",
        description="""Iterative RLM execution loop implementing recursive decomposition.

        Each iteration:
        1. code_generator: Generates Python code using llm_query() for sub-problems
        2. code_executor: Executes code in persistent REPL, updates iteration_history
        3. completion_checker: Detects FINAL/FINAL_VAR and escalates to exit

        Callbacks handle:
        - State initialization and cleanup
        - Metrics tracking
        - Error recovery
        """,
        max_iterations=max_iterations,
        sub_agents=[
            make_code_generator(),
            make_code_executor(),
            RLMCompletionChecker(),
        ],
        before_agent_callback=callbacks.get("before_agent_callback"),
        after_agent_callback=callbacks.get("after_agent_callback"),
    )


def make_rlm_completion_workflow(max_iterations: int = 10) -> SequentialAgent:
    """Create the full RLM completion workflow.

    Args:
        max_iterations: Maximum iterations for the loop.

    Returns:
        SequentialAgent for complete RLM workflow.
    """
    return SequentialAgent(
        name="rlm_completion_workflow",
        description="""Full RLM recursive decomposition workflow.

        Complete pipeline for analyzing large datasets using the RLM paradigm:
        1. Loads context from Unity Catalog or custom sources
        2. Iteratively generates and executes code with llm_query()
        3. Continues until FINAL answer is produced
        4. Formats results for user presentation

        Uses composed system prompt from rlm_adk/prompts.py
        with healthcare vendor management extensions.
        """,
        sub_agents=[
            make_context_setup_agent(),
            make_rlm_iteration_loop(max_iterations),
            make_result_formatter(),
        ],
    )
