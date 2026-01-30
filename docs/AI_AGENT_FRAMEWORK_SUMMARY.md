# AI Agent Framework Summary

## Framework Type: Custom Multi-Agent System

This project uses a **custom-built multi-agent architecture** powered by **Google Gemini** (via `google-genai` SDK). It does NOT use popular agent frameworks like LangChain, LangGraph, CrewAI, or AutoGen.

---

## Core Architecture: 3-Phase Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     OrchestratorAgent                           │
│               (Coordinates all agent execution)                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
  ┌───────────┐      ┌───────────┐      ┌───────────┐
  │  Phase 1  │      │  Phase 2  │      │  Phase 3  │
  │  Market   │─────▶│ Portfolio │─────▶│  Summary  │
  │Intelligence│     │  Insight  │      │Generation │
  └───────────┘      └───────────┘      └───────────┘
```

---

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `BaseAgent` | `app/agents/base.py` | Abstract base class with retry logic, tracing, tool support |
| `OrchestratorAgent` | `app/agents/orchestrator.py` | Coordinates the 3-phase pipeline |
| `VertexAIClient` | `app/utils/vertex_ai_client.py` | Wrapper for Google Gemini API |
| Specialized Agents | `app/agents/*.py` | MarketIntelligence, PortfolioInsight, SummaryGeneration |

---

## Key Design Patterns

### 1. Generic Type-Safe Agents

```python
class BaseAgent(ABC, Generic[InputSchema, OutputSchema]):
    input_schema: type[InputSchema]   # Pydantic model
    output_schema: type[OutputSchema] # Pydantic model
```

### 2. Configuration-Driven

```python
AgentConfig(
    name="market_intelligence_agent",
    model_name="gemini-2.0-flash-exp",
    temperature=0.1,
    timeout_seconds=30,
    retry_attempts=2,
)
```

### 3. Built-in Retry with Backoff

- Automatic retry on failures (configurable attempts)
- Exponential backoff between retries
- Timeout handling with `asyncio.wait_for()`

### 4. Graceful Degradation

- Phase 1 (Market Intelligence) failure → Return fallback response
- Phase 2/3 failure → Continue with defaults, set `degraded_mode=True`

---

## LLM Provider

| Attribute | Value |
|-----------|-------|
| **Provider** | Google AI (Gemini) |
| **SDK** | `google-genai` (version >= 1.0.0) |
| **Default Model** | `gemini-2.0-flash-exp` |
| **Features** | JSON-mode responses, tool/function calling support |

---

## Why Custom vs. LangChain/CrewAI?

| Aspect | This Framework | LangChain/CrewAI |
|--------|----------------|------------------|
| Simplicity | Minimal dependencies | Heavy dependencies |
| Type Safety | Native Pydantic generics | Varies |
| Control | Full control over flow | Framework constraints |
| Learning Curve | Read `base.py` (400 lines) | Learn entire ecosystem |
| Overhead | Lightweight | Significant abstraction layers |

---

## How to Create a New Agent

1. Define Pydantic schemas for input/output in `app/models/agent_schemas.py`
2. Create agent class extending `BaseAgent[InputSchema, OutputSchema]`
3. Implement `get_system_prompt()` and `execute()` methods
4. Register in orchestrator if needed

### Example

```python
from app.agents.base import BaseAgent, AgentConfig, AgentExecutionContext
from app.models.agent_schemas import MyInput, MyOutput

class MyAgent(BaseAgent[MyInput, MyOutput]):
    input_schema = MyInput
    output_schema = MyOutput
    
    def __init__(self):
        config = AgentConfig(
            name="my_agent",
            description="What this agent does",
            model_name="gemini-2.0-flash-exp",
            temperature=0.1,
        )
        super().__init__(config)
    
    def get_system_prompt(self) -> str:
        return "You are an expert at..."
    
    async def execute(
        self,
        input_data: MyInput,
        context: AgentExecutionContext,
    ) -> MyOutput:
        # Agent logic here
        prompt = self.build_context_prompt(input_data)
        response = await self.generate_with_model(prompt)
        return self.parse_response(response, MyOutput)
```

---

## Key Files to Read

| Priority | File | What You'll Learn |
|----------|------|-------------------|
| 1 | `app/agents/base.py` | Core agent contract & patterns |
| 2 | `app/agents/orchestrator.py` | How agents are coordinated |
| 3 | `app/agents/AGENTS_GUIDE.md` | Complete documentation |
| 4 | `app/utils/vertex_ai_client.py` | LLM integration details |

---

## Data Flow Between Agents

```
MarketPulseRequest
        │
        ▼
┌───────────────────────────────────────────────────────┐
│            Phase 1: MarketIntelligenceAgent           │
│  INPUT: selected_indices, time_window_hours           │
│  OUTPUT: market_phase, indices_data, news_items,      │
│          preliminary_themes, market_outlook           │
└───────────────────────────┬───────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────┐
│            Phase 2: PortfolioInsightAgent             │
│  INPUT: user_id + Phase 1 outputs                     │
│  OUTPUT: news_with_impacts, portfolio_impact,         │
│          watchlist_alerts, refined_themes             │
└───────────────────────────┬───────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────┐
│           Phase 3: SummaryGenerationAgent             │
│  INPUT: Phase 1 + Phase 2 outputs                     │
│  OUTPUT: market_summary_bullets, executive_summary,   │
│          key_takeaways, trending_now_section          │
└───────────────────────────┬───────────────────────────┘
                            │
                            ▼
                  MarketPulseResponse
```

---

## Error Handling

| Failed Agent | Criticality | Behavior |
|--------------|-------------|----------|
| `MarketIntelligenceAgent` | **Critical** | Cannot proceed, return fallback response |
| `PortfolioInsightAgent` | Non-critical | Continue with defaults, `degraded_mode=True` |
| `SummaryGenerationAgent` | Non-critical | Use basic summary, `degraded_mode=True` |

---

## Quick Reference: Agent Responsibilities

| Agent | Gets | Produces | Key Feature |
|-------|------|----------|-------------|
| `MarketIntelligenceAgent` | Indices, time range | Market data, news, themes | Determines market phase |
| `PortfolioInsightAgent` | User ID, news | Portfolio impact, alerts | Causal chain analysis |
| `SummaryGenerationAgent` | All prior data | Human-readable summaries | Causal language |
| `OrchestratorAgent` | Request | Final response | Phase coordination |

---

*For detailed implementation guidance, see `app/agents/AGENTS_GUIDE.md`*
