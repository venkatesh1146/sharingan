"""
Tool Registry for the Multi-Agent System.

Provides a central registry for all tools available to agents,
enabling easy tool discovery and registration.
"""

from typing import Any, Callable, Dict, List, Optional

from google.generativeai.types import Tool

# TODO: Create market_data_tools.py with get_market_data_tools() and get_market_data_tool_handlers()
# from app.tools.market_data_tools import (
#     get_market_data_tools,
#     get_market_data_tool_handlers,
# )
# from app.tools.news_tools import (
#     get_news_tools,
#     get_news_tool_handlers,
# )
# TODO: Create user_data_tools.py with get_user_data_tools() and get_user_data_tool_handlers()
# from app.tools.user_data_tools import (
#     get_user_data_tools,
#     get_user_data_tool_handlers,
# )
from app.tools.analysis_tools import (
    get_analysis_tools,
    get_analysis_tool_handlers,
)


class ToolRegistry:
    """
    Central registry for all agent tools.
    
    Provides methods to:
    - Get tools by category
    - Get tool handlers by category
    - Get all tools combined
    - Register custom tools
    """

    def __init__(self):
        self._custom_tools: List[Tool] = []
        self._custom_handlers: Dict[str, Callable] = {}

    # def get_market_data_tools(self) -> List[Tool]:
    #     """Get tools for Market Data Agent."""
    #     return get_market_data_tools()

    # def get_market_data_handlers(self) -> Dict[str, Callable]:
    #     """Get tool handlers for Market Data Agent."""
    #     return get_market_data_tool_handlers()

    # def get_news_tools(self) -> List[Tool]:
    #     """Get tools for News Analysis Agent."""
    #     return get_news_tools()

    # def get_news_handlers(self) -> Dict[str, Callable]:
    #     """Get tool handlers for News Analysis Agent."""
    #     return get_news_tool_handlers()

    # def get_user_data_tools(self) -> List[Tool]:
    #     """Get tools for User Context Agent."""
    #     return get_user_data_tools()

    # def get_user_data_handlers(self) -> Dict[str, Callable]:
    #     """Get tool handlers for User Context Agent."""
    #     return get_user_data_tool_handlers()

    def get_analysis_tools(self) -> List[Tool]:
        """Get tools for Impact Analysis Agent."""
        return get_analysis_tools()

    def get_analysis_handlers(self) -> Dict[str, Callable]:
        """Get tool handlers for Impact Analysis Agent."""
        return get_analysis_tool_handlers()

    def get_all_tools(self) -> List[Tool]:
        """
        Get all registered tools.
        
        Returns:
            Combined list of all tools from all categories
        """
        all_tools = (
            # self.get_market_data_tools() +
            # self.get_news_tools() +
            # self.get_user_data_tools() +
            self.get_analysis_tools() +
            self._custom_tools
        )
        return all_tools

    def get_all_handlers(self) -> Dict[str, Callable]:
        """
        Get all tool handlers combined.
        
        Returns:
            Dictionary mapping tool names to handlers
        """
        all_handlers = {}
        # all_handlers.update(self.get_market_data_handlers())
        # all_handlers.update(self.get_news_handlers())
        # all_handlers.update(self.get_user_data_handlers())
        all_handlers.update(self.get_analysis_handlers())
        all_handlers.update(self._custom_handlers)
        return all_handlers

    def register_tool(
        self,
        tool: Tool,
        handlers: Dict[str, Callable],
    ) -> None:
        """
        Register a custom tool with its handlers.
        
        Args:
            tool: Vertex AI Tool object
            handlers: Dictionary mapping function names to handlers
        """
        self._custom_tools.append(tool)
        self._custom_handlers.update(handlers)

    def get_tools_for_agent(
        self,
        agent_name: str,
    ) -> tuple[List[Tool], Dict[str, Callable]]:
        """
        Get tools appropriate for a specific agent.
        
        Args:
            agent_name: Name of the agent
        
        Returns:
            Tuple of (tools list, handlers dict)
        """
        agent_tools_map = {
            "market_data_agent": (
                self.get_market_data_tools,
                self.get_market_data_handlers,
            ),
            "news_analysis_agent": (
                self.get_news_tools,
                self.get_news_handlers,
            ),
            "user_context_agent": (
                self.get_user_data_tools,
                self.get_user_data_handlers,
            ),
            "impact_analysis_agent": (
                self.get_analysis_tools,
                self.get_analysis_handlers,
            ),
            "summary_generation_agent": (
                # Summary agent uses analysis tools for ranking
                self.get_analysis_tools,
                self.get_analysis_handlers,
            ),
        }

        if agent_name in agent_tools_map:
            tools_fn, handlers_fn = agent_tools_map[agent_name]
            return tools_fn(), handlers_fn()

        # Default: return all tools
        return self.get_all_tools(), self.get_all_handlers()


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.
    
    Uses singleton pattern to ensure single registry across the application.
    """
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
