"""
Google AI client wrapper for the Multi-Agent system.

Provides a unified interface for interacting with Google AI (Gemini)
generative models, with support for:
- Tool/function calling
- Retry logic
- Response parsing
- Error handling
"""

import json
from typing import Any, Dict, List, Optional, Callable, TypeVar
import asyncio

from google import genai
from google.genai import types

from app.config import get_settings
from app.utils.logging import get_logger
from app.utils.exceptions import AgentReasoningError

logger = get_logger(__name__)

T = TypeVar("T")

# Global client instance
_client: Optional[genai.Client] = None


def get_genai_client() -> genai.Client:
    """Get or create the global Google AI client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = genai.Client(api_key=settings.GOOGLE_AI_API_KEY)
    return _client


class VertexAIClient:
    """
    Client wrapper for Google AI (Gemini) generative models.
    
    Handles model initialization, configuration, and provides
    methods for chat-based interactions with tool calling support.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
        tools: Optional[List[Any]] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize the Google AI client.
        
        Args:
            model_name: Model to use (defaults to GEMINI_FAST_MODEL)
            temperature: Generation temperature (0-1)
            max_output_tokens: Maximum tokens in response
            tools: List of tools for function calling
            system_instruction: System prompt for the model
        """
        settings = get_settings()
        
        self.client = get_genai_client()
        self.model_name = model_name or settings.GEMINI_FAST_MODEL
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.tools = tools
        self.system_instruction = system_instruction

        # Create generation config
        self.generation_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json" if not tools else None,
            system_instruction=system_instruction,
        )

    async def generate_content(
        self,
        prompt: str,
        **kwargs,
    ) -> str:
        """
        Generate content from a prompt (single turn).
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text response
        """
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=self.generation_config,
            )

            if response.text:
                return response.text

            raise AgentReasoningError(
                agent_name="google_ai",
                message="No text in response",
            )

        except Exception as e:
            logger.error("google_ai_generation_error", error=str(e))
            raise

    async def chat_with_tools(
        self,
        prompt: str,
        tool_handlers: Dict[str, Callable],
        max_turns: int = 10,
    ) -> str:
        """
        Execute a chat session with tool calling support.
        
        This method handles multi-turn conversations where the model
        can call tools (functions) and receive their results.
        
        Args:
            prompt: Initial user prompt
            tool_handlers: Dict mapping tool names to handler functions
            max_turns: Maximum conversation turns to prevent infinite loops
        
        Returns:
            Final text response from the model
        """
        # For now, use simple generation without tools
        # Tool calling in google-genai requires different setup
        return await self.generate_content(prompt)

    async def _execute_tool_handler(
        self,
        handler: Callable,
        args: Dict[str, Any],
    ) -> Any:
        """Execute a tool handler, handling both sync and async functions."""
        if asyncio.iscoroutinefunction(handler):
            return await handler(**args)
        else:
            return await asyncio.to_thread(handler, **args)

    def parse_json_response(
        self,
        response_text: str,
        expected_type: type[T],
    ) -> T:
        """
        Parse a JSON response into a Pydantic model.
        
        Args:
            response_text: Raw JSON text from model
            expected_type: Pydantic model class to parse into
        
        Returns:
            Parsed and validated model instance
        """
        try:
            # Handle markdown code blocks
            text = response_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]

            data = json.loads(text.strip())
            return expected_type(**data)

        except json.JSONDecodeError as e:
            logger.error(
                "json_parse_error",
                error=str(e),
                response_preview=response_text[:500],
            )
            raise AgentReasoningError(
                agent_name="google_ai",
                message=f"Invalid JSON in response: {str(e)}",
                raw_response=response_text,
            )
        except Exception as e:
            logger.error(
                "response_validation_error",
                error=str(e),
                expected_type=expected_type.__name__,
            )
            raise
