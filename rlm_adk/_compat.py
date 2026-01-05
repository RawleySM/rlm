"""Compatibility layer for optional google-adk dependency.

This module provides type definitions and compatibility shims that allow
the tools to work both with and without google-adk installed. When running
tests or using tools standalone, a simple mock context is used. When running
with the full ADK runtime, the actual ToolContext is used.
"""

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

# Check if google-adk is available
try:
    from google.adk.tools import ToolContext as ADKToolContext

    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    ADKToolContext = None


@runtime_checkable
class ToolContextProtocol(Protocol):
    """Protocol defining the expected interface for tool context.

    This allows tools to work with both:
    - The real google.adk.tools.ToolContext when ADK is installed
    - A simple mock context for testing without ADK
    """

    @property
    def state(self) -> dict[str, Any]:
        """Access the session state dictionary."""
        ...


class SimpleToolContext:
    """Simple tool context for use when google-adk is not installed.

    Provides a minimal implementation of the ToolContext interface
    for testing and standalone tool usage.
    """

    def __init__(self, state: dict[str, Any] | None = None):
        self._state = state or {}

    @property
    def state(self) -> dict[str, Any]:
        """Access the session state dictionary."""
        return self._state


# Export the appropriate type based on availability
if TYPE_CHECKING:
    # For type checking, use the protocol
    ToolContext = ToolContextProtocol
else:
    # At runtime, try to use the real ToolContext, fall back to protocol
    if ADK_AVAILABLE:
        from google.adk.tools import ToolContext
    else:
        ToolContext = ToolContextProtocol


def create_tool_context(state: dict[str, Any] | None = None) -> ToolContextProtocol:
    """Create a tool context for testing or standalone use.

    Args:
        state: Optional initial state dictionary.

    Returns:
        A ToolContext-compatible object.
    """
    return SimpleToolContext(state)


def check_adk_available() -> None:
    """Raise ImportError if google-adk is not installed.

    Call this at the start of modules that require the full ADK runtime.
    """
    if not ADK_AVAILABLE:
        raise ImportError(
            "google-adk is required for RLM-ADK agents. "
            "Install it with: pip install google-adk "
            "or: pip install rlm[rlm-adk]"
        )


__all__ = [
    "ADK_AVAILABLE",
    "ToolContext",
    "ToolContextProtocol",
    "SimpleToolContext",
    "create_tool_context",
    "check_adk_available",
]
