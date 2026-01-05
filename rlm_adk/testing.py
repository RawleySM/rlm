"""Testing utilities for RLM-ADK."""

from typing import Any


class MockToolContext:
    """Mock tool context for testing.

    Provides a minimal implementation compatible with ToolContext interface
    used by RLM-ADK tools (accessing .state).
    """

    def __init__(self, state: dict[str, Any] | None = None):
        self._state = state or {}

    @property
    def state(self) -> dict[str, Any]:
        """Access the session state dictionary."""
        return self._state


def create_tool_context(state: dict[str, Any] | None = None) -> MockToolContext:
    """Create a mock tool context for testing.

    Args:
        state: Optional initial state dictionary.

    Returns:
        A MockToolContext object.
    """
    return MockToolContext(state)

