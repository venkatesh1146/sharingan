"""
Base Agent class for the Multi-Agent System.

Provides the abstract foundation for all specialized agents with:
- Type-safe input/output schemas
- Execution with retry logic
- Tool calling support
- Logging and tracing
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar
import asyncio
import time
import json

from pydantic import BaseModel

from app.config import get_settings
from app.utils.logging import AgentLogger
from app.utils.tracing import trace_agent_execution
from app.utils.exceptions import (
    AgentExecutionError,
    AgentTimeoutError,
    AgentReasoningError,
    DataValidationError,
)
from app.utils.vertex_ai_client import VertexAIClient
from app.models.agent_schemas import AgentExecutionResult


# Type variables for generic input/output schemas
InputSchema = TypeVar("InputSchema", bound=BaseModel)
OutputSchema = TypeVar("OutputSchema", bound=BaseModel)


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    name: str
    description: str = ""
    model_name: str = "gemini-2.0-flash-exp"
    temperature: float = 0.1
    max_output_tokens: int = 4096
    timeout_seconds: int = 30
    retry_attempts: int = 2
    retry_delay_seconds: float = 1.0

    class Config:
        arbitrary_types_allowed = True


class AgentExecutionContext(BaseModel):
    """Context passed to agent during execution."""

    request_id: str
    user_id: str
    timestamp: datetime = datetime.utcnow()
    trace_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True


class BaseAgent(ABC, Generic[InputSchema, OutputSchema]):
    """
    Abstract base class for all agents in the Multi-Agent system.
    
    Each agent must:
    1. Define input/output schemas (via type parameters)
    2. Implement get_system_prompt() to define agent behavior
    3. Implement execute() for main logic
    4. Register required tools via get_tools()
    
    The base class provides:
    - Retry logic with exponential backoff
    - Execution timing and metrics
    - Logging with agent context
    - Tool calling support via Google AI
    - Response parsing and validation
    """

    # Subclasses should override these type hints
    input_schema: type[InputSchema]
    output_schema: type[OutputSchema]

    def __init__(self, config: AgentConfig):
        """
        Initialize the agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.logger = AgentLogger(config.name)
        self._client: Optional[VertexAIClient] = None
        self._tool_handlers: Dict[str, Callable] = {}

    @property
    def client(self) -> VertexAIClient:
        """
        Lazy initialization of Google AI client.
        
        This allows agents to be instantiated without immediately
        connecting to Google AI.
        """
        if self._client is None:
            self._client = VertexAIClient(
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                tools=self.get_tools(),
                system_instruction=self.get_system_prompt(),
            )
        return self._client

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return the system prompt for this agent.
        
        The system prompt defines the agent's role, capabilities,
        and constraints. It should be specific and focused on
        the agent's single responsibility.
        """
        pass

    @abstractmethod
    async def execute(
        self,
        input_data: InputSchema,
        context: AgentExecutionContext,
    ) -> OutputSchema:
        """
        Execute the agent's main logic.
        
        This method contains the core reasoning and tool calling
        logic for the agent. It should:
        1. Process input data
        2. Call necessary tools
        3. Use Vertex AI for reasoning
        4. Return validated output
        
        Args:
            input_data: Validated input data
            context: Execution context
        
        Returns:
            Validated output data
        
        Raises:
            AgentExecutionError: On execution failures
            AgentReasoningError: On invalid model output
        """
        pass

    def get_tools(self) -> Optional[List[Any]]:
        """
        Return the list of tools available to this agent.
        
        Override this method to provide tools for function calling.
        By default, returns None (no tools).
        """
        return None

    def get_tool_handlers(self) -> Dict[str, Callable]:
        """
        Return mapping of tool names to handler functions.
        
        Override this method to provide handlers for tools.
        Handlers can be sync or async functions.
        """
        return self._tool_handlers

    def register_tool_handler(self, name: str, handler: Callable) -> None:
        """Register a handler for a specific tool."""
        self._tool_handlers[name] = handler

    async def execute_with_retry(
        self,
        input_data: InputSchema,
        context: AgentExecutionContext,
    ) -> AgentExecutionResult:
        """
        Execute the agent with retry logic and error handling.
        
        This is the primary method to call when running an agent.
        It wraps execute() with:
        - Timing metrics
        - Retry logic
        - Error handling
        - Logging
        - Tracing
        
        Args:
            input_data: Input data for the agent
            context: Execution context
        
        Returns:
            AgentExecutionResult with status, output, and metrics
        """
        start_time = time.time()
        last_error: Optional[Exception] = None

        for attempt in range(self.config.retry_attempts):
            try:
                self.logger.execution_start(
                    request_id=context.request_id,
                    attempt=attempt + 1,
                    max_attempts=self.config.retry_attempts,
                )

                # Execute with timeout and tracing
                with trace_agent_execution(
                    agent_name=self.config.name,
                    request_id=context.request_id,
                    user_id=context.user_id,
                    attempt=attempt + 1,
                ):
                    output = await asyncio.wait_for(
                        self.execute(input_data, context),
                        timeout=self.config.timeout_seconds,
                    )

                execution_time_ms = int((time.time() - start_time) * 1000)

                self.logger.execution_success(
                    request_id=context.request_id,
                    execution_time_ms=execution_time_ms,
                )

                return AgentExecutionResult(
                    agent_name=self.config.name,
                    status="success",
                    output=output,
                    execution_time_ms=execution_time_ms,
                    retry_count=attempt,
                )

            except asyncio.TimeoutError:
                last_error = AgentTimeoutError(
                    agent_name=self.config.name,
                    timeout_seconds=self.config.timeout_seconds,
                )
                self.logger.warning(
                    "agent_timeout",
                    attempt=attempt + 1,
                    timeout_seconds=self.config.timeout_seconds,
                )

            except AgentReasoningError as e:
                last_error = e
                self.logger.warning(
                    "agent_reasoning_error",
                    attempt=attempt + 1,
                    error=str(e),
                )

            except Exception as e:
                last_error = AgentExecutionError(
                    agent_name=self.config.name,
                    message=str(e),
                )
                self.logger.error(
                    "agent_execution_error",
                    attempt=attempt + 1,
                    error=str(e),
                    error_type=type(e).__name__,
                )

            # Wait before retry (except on last attempt)
            if attempt < self.config.retry_attempts - 1:
                await asyncio.sleep(
                    self.config.retry_delay_seconds * (attempt + 1)
                )

        # All retries exhausted
        execution_time_ms = int((time.time() - start_time) * 1000)

        self.logger.execution_failure(
            request_id=context.request_id,
            error=str(last_error),
        )

        return AgentExecutionResult(
            agent_name=self.config.name,
            status="failed",
            error=str(last_error),
            execution_time_ms=execution_time_ms,
            retry_count=self.config.retry_attempts,
        )

    async def generate_with_model(
        self,
        prompt: str,
        use_tools: bool = True,
    ) -> str:
        """
        Generate content using the Vertex AI model.
        
        Args:
            prompt: The prompt to send to the model
            use_tools: Whether to enable tool calling
        
        Returns:
            Generated text response
        """
        if use_tools and self.get_tools():
            return await self.client.chat_with_tools(
                prompt=prompt,
                tool_handlers=self.get_tool_handlers(),
            )
        else:
            return await self.client.generate_content(prompt)

    def parse_response(
        self,
        response_text: str,
        output_class: type[OutputSchema],
    ) -> OutputSchema:
        """
        Parse and validate model response into output schema.
        
        Args:
            response_text: Raw text response from model
            output_class: Pydantic model class for validation
        
        Returns:
            Validated output instance
        
        Raises:
            AgentReasoningError: If parsing or validation fails
        """
        try:
            # Clean up response text
            text = response_text.strip()

            # Handle markdown code blocks
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]

            # Parse JSON
            data = json.loads(text.strip())

            # Validate with Pydantic
            return output_class(**data)

        except json.JSONDecodeError as e:
            self.logger.error(
                "json_parse_error",
                error=str(e),
                response_preview=response_text[:500],
            )
            raise AgentReasoningError(
                agent_name=self.config.name,
                message=f"Invalid JSON in response: {e}",
                raw_response=response_text,
            )

        except Exception as e:
            self.logger.error(
                "validation_error",
                error=str(e),
                expected_type=output_class.__name__,
            )
            raise DataValidationError(
                message=f"Output validation failed for {self.config.name}: {e}",
            )

    def build_context_prompt(self, input_data: InputSchema) -> str:
        """
        Build a context-rich prompt from input data.
        
        Override this to customize how input data is formatted
        into a prompt string.
        """
        return json.dumps(input_data.model_dump(), indent=2, default=str)

    def validate_input(self, input_data: InputSchema) -> None:
        """
        Validate input data before execution.
        
        Override to add custom validation logic.
        """
        pass

    def validate_output(self, output_data: OutputSchema) -> None:
        """
        Validate output data after execution.
        
        Override to add custom validation logic.
        """
        pass
