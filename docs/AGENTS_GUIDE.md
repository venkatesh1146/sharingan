# Multi-Agent System Architecture Guide

## Overview

This guide explains how the agents in the `app/agents/` directory work together to power the Market Pulse feature. The system uses a **3-agent architecture** coordinated by an orchestrator to generate personalized market insights.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Agent Hierarchy](#agent-hierarchy)
3. [Execution Flow](#execution-flow)
4. [Base Agent Framework](#base-agent-framework)
5. [Specialized Agents](#specialized-agents)
6. [Data Flow Between Agents](#data-flow-between-agents)
7. [Error Handling & Degraded Mode](#error-handling--degraded-mode)
8. [Adding a New Agent](#adding-a-new-agent)
9. [Best Practices](#best-practices)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        OrchestratorAgent                              │
│                  (Coordinates all agent execution)                    │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
          ▼                     ▼                     ▼
   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
   │   Phase 1    │     │   Phase 2    │     │   Phase 3    │
   │   Market     │────▶│  Portfolio   │────▶│   Summary    │
   │ Intelligence │     │   Insight    │     │  Generation  │
   └──────────────┘     └──────────────┘     └──────────────┘
         │                     │                     │
         │                     │                     │
         ▼                     ▼                     ▼
   - Market Data        - User Watchlist      - Causal Summaries
   - Market Phase       - User Portfolio      - Trending News
   - News Analysis      - Impact Analysis     - Executive Summary
   - Theme Clustering   - Watchlist Alerts    - Key Takeaways
```

**Key Design Principles:**
- **Sequential phases** - Each agent depends on the previous phase's output
- **Type-safe contracts** - Pydantic schemas ensure data integrity between agents
- **Fault tolerance** - Graceful degradation when agents fail
- **Single responsibility** - Each agent has a focused purpose

---

## Agent Hierarchy

### File Structure

```
app/agents/
├── __init__.py                    # Exports all agents
├── base.py                        # BaseAgent abstract class
├── orchestrator.py                # OrchestratorAgent (coordinator)
├── market_intelligence_agent.py   # Phase 1: Market data + news
├── portfolio_insight_agent.py     # Phase 2: User context + impact
└── summary_generation_agent.py    # Phase 3: Summary generation
```

### Inheritance

```
BaseAgent[InputSchema, OutputSchema]
    ├── MarketIntelligenceAgent
    ├── PortfolioInsightAgent
    └── SummaryGenerationAgent

OrchestratorAgent (standalone, not inheriting from BaseAgent)
```

---

## Execution Flow

### Phase-by-Phase Breakdown

```python
# 1. User sends MarketPulseRequest
request = MarketPulseRequest(
    user_id="user123",
    selected_indices=["NIFTY 50", "SENSEX"],
    include_portfolio=True,
    include_watchlist=True
)

# 2. Orchestrator starts 3-phase flow
orchestrator = OrchestratorAgent()
response = await orchestrator.orchestrate(request, context)
```

| Phase | Agent | Input From | Output To | Purpose |
|-------|-------|------------|-----------|---------|
| 1 | `MarketIntelligenceAgent` | Request | Phase 2 | Fetch market data, analyze news, determine market phase |
| 2 | `PortfolioInsightAgent` | Phase 1 + Request | Phase 3 | Fetch user data, analyze impact on holdings |
| 3 | `SummaryGenerationAgent` | Phase 1 + Phase 2 | Final Response | Generate human-readable summaries |

### Sequence Diagram

```
User Request
     │
     ▼
┌────────────────────┐
│  OrchestratorAgent │
└────────┬───────────┘
         │
         ▼ Phase 1
┌────────────────────────┐
│ MarketIntelligenceAgent│
│  • Fetch indices data  │
│  • Determine phase     │
│  • Fetch & analyze news│
│  • Cluster themes      │
└────────┬───────────────┘
         │ Output: market_phase, indices_data, news_items, themes
         ▼ Phase 2
┌────────────────────────┐
│ PortfolioInsightAgent  │
│  • Fetch watchlist     │
│  • Fetch portfolio     │
│  • Analyze impacts     │
│  • Generate alerts     │
└────────┬───────────────┘
         │ Output: portfolio, impacts, alerts, refined_themes
         ▼ Phase 3
┌────────────────────────┐
│ SummaryGenerationAgent │
│  • Generate bullets    │
│  • Create trending sec │
│  • Executive summary   │
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  MarketPulseResponse   │
└────────────────────────┘
```

---

## Base Agent Framework

The `BaseAgent` class (`base.py`) provides the foundation for all specialized agents.

### Key Components

#### 1. Type-Safe Generic Pattern

```python
class BaseAgent(ABC, Generic[InputSchema, OutputSchema]):
    """
    InputSchema and OutputSchema are Pydantic models.
    Each agent defines these for compile-time type checking.
    """
    input_schema: type[InputSchema]
    output_schema: type[OutputSchema]
```

#### 2. Agent Configuration

```python
class AgentConfig(BaseModel):
    name: str                    # Unique identifier
    description: str             # Human-readable description
    model_name: str              # e.g., "gemini-2.0-flash-exp"
    temperature: float           # 0.0-1.0, lower = more deterministic
    max_output_tokens: int       # Max response length
    timeout_seconds: int         # Execution timeout
    retry_attempts: int          # Auto-retry count
    retry_delay_seconds: float   # Delay between retries
```

#### 3. Execution Context

```python
class AgentExecutionContext(BaseModel):
    request_id: str      # Unique request identifier
    user_id: str         # User making the request
    timestamp: datetime  # When the request was made
    trace_id: str        # For distributed tracing
    metadata: Dict       # Additional context
```

### Abstract Methods (Must Implement)

| Method | Purpose |
|--------|---------|
| `get_system_prompt()` | Returns the LLM system prompt defining agent behavior |
| `execute(input_data, context)` | Main execution logic, returns OutputSchema |

### Built-in Methods (Free to Use)

| Method | Purpose |
|--------|---------|
| `execute_with_retry()` | Wraps `execute()` with retry logic, timing, tracing |
| `generate_with_model()` | Call Vertex AI with optional tool calling |
| `parse_response()` | Parse LLM JSON response into Pydantic model |
| `get_tools()` | Override to provide function-calling tools |
| `register_tool_handler()` | Register handlers for tool calls |

### Example: How Retry Works

```python
# Called by orchestrator:
result = await agent.execute_with_retry(input_data, context)

# Internally:
# 1. Logs start of execution
# 2. Wraps in timeout (asyncio.wait_for)
# 3. Wraps in tracing span
# 4. On failure, retries up to config.retry_attempts times
# 5. Returns AgentExecutionResult with status, output, timing
```

---

## Specialized Agents

### 1. MarketIntelligenceAgent

**Purpose:** Combines market data fetching and news analysis in a single agent.

**File:** `market_intelligence_agent.py`

**Input Schema:**
```python
class MarketIntelligenceAgentInput:
    selected_indices: List[str]    # ["NIFTY 50", "SENSEX"]
    timestamp: datetime
    force_refresh: bool
    time_window_hours: int         # 1-72 hours of news
    max_articles: int              # 10-100 articles
    watchlist: Optional[List[str]] # For stock-specific news
```

**Output Schema:**
```python
class MarketIntelligenceAgentOutput:
    # Market data
    market_phase: Literal["pre", "mid", "post"]
    indices_data: Dict[str, IndexData]
    market_outlook: Optional[MarketOutlook]  # None during mid-market
    market_momentum: str
    
    # News analysis
    news_items: List[NewsItem]
    sentiment_distribution: Dict[str, int]
    preliminary_themes: List[PreliminaryTheme]
    key_topics: List[str]
    breaking_news: List[str]
```

**Execution Steps:**
1. Fetch market intelligence data (indices + news) via service
2. Determine market phase (pre/mid/post based on time)
3. Process indices data into `IndexData` objects
4. Calculate market outlook (only for pre/post market)
5. Get market momentum indicator
6. Process news items with sentiment
7. Calculate sentiment distribution
8. Cluster news into preliminary themes

**Key Methods:**
```python
def _generate_outlook_reasoning()  # Human-readable outlook explanation
def _identify_key_drivers()        # What's moving the market
def _calculate_sentiment_distribution()  # bullish/bearish/neutral counts
def _extract_key_topics()          # Main topics from themes
```

---

### 2. PortfolioInsightAgent

**Purpose:** Combines user context fetching and impact analysis.

**File:** `portfolio_insight_agent.py`

**Input Schema:**
```python
class PortfolioInsightAgentInput:
    user_id: str
    include_watchlist: bool
    include_portfolio: bool
    news_filter: Literal["all", "watchlist", "portfolio"]
    
    # From Phase 1
    news_items: List[NewsItem]
    indices_data: Dict[str, IndexData]
    preliminary_themes: List[PreliminaryTheme]
```

**Output Schema:**
```python
class PortfolioInsightAgentOutput:
    # User context
    user_id: str
    watchlist: List[str]
    portfolio: List[PortfolioHolding]
    sector_exposure: Dict[str, float]
    total_portfolio_value: float
    risk_profile: Optional[str]
    
    # Impact analysis
    news_with_impacts: List[NewsWithImpact]
    portfolio_level_impact: PortfolioImpact
    watchlist_alerts: List[WatchlistAlert]
    refined_themes: List[ThemeGroup]
    sector_impact_summary: Dict[str, str]
    causal_chains: List[str]
```

**Execution Steps:**
1. **Phase 1 - Fetch User Context:**
   - Fetch watchlist via `fetch_user_watchlist()`
   - Fetch portfolio via `fetch_user_portfolio()`
   - Calculate sector exposure
   - Get user preferences

2. **Phase 2 - Impact Analysis:**
   - Analyze each news item for stock impacts
   - Build causal chains (event → impact → stock)
   - Determine sector-level impacts
   - Calculate portfolio-level impact
   - Generate watchlist alerts
   - Refine themes with user context

**Key Methods:**
```python
async def _analyze_news_impact()       # Analyze single news for impacts
def _build_causal_chain()              # "Oil prices ↑ → Negative → ONGC"
def _calculate_portfolio_impact()      # Aggregate impact on holdings
def _generate_watchlist_alerts()       # Alerts for stocks user watches
def _refine_themes()                   # Add user relevance to themes
```

**Causal Chain Example:**
```python
# Input: News about oil price surge
# Output: "Oil prices surge → Negative pressure → Airlines sector"

# The agent identifies supply chain relationships:
supply_chain_events = ["oil", "crude", "steel", "rupee", "dollar", "rate"]
```

---

### 3. SummaryGenerationAgent

**Purpose:** Generates human-readable summaries with causal language.

**File:** `summary_generation_agent.py`

**Input Schema:**
```python
class SummaryGenerationAgentInput:
    market_outlook: Optional[MarketOutlook]
    market_phase: Literal["pre", "mid", "post"]
    news_with_impacts: List[NewsWithImpact]
    indices_data: Dict[str, IndexData]
    portfolio_impact: PortfolioImpact
    refined_themes: List[ThemeGroup]
    max_bullets: int  # 1-5
```

**Output Schema:**
```python
class SummaryGenerationAgentOutput:
    market_summary_bullets: Optional[List[MarketSummaryBullet]]  # Pre/post only
    trending_now_section: Optional[List[NewsItem]]              # Mid-market only
    executive_summary: str
    key_takeaways: List[str]
    generation_metadata: Dict
```

**Market Phase Behavior:**
| Phase | Output |
|-------|--------|
| `pre` / `post` | Market summary bullets with causal language |
| `mid` | Trending news section (no outlook available) |

**Causal Language Requirement:**

All summary bullets **must** contain causal keywords:
```python
CAUSAL_KEYWORDS = [
    "due to", "after", "following", "driven by", "as", "because",
    "on account of", "amid", "on the back of", "triggered by",
    "led by", "supported by", "weighed by"
]
```

**Example Bullets:**
```
✅ "Banking stocks gained driven by RBI's rate cut announcement."
✅ "IT sector under pressure following weak US tech earnings."
❌ "Banking stocks are up today." (No causal explanation)
```

---

## Data Flow Between Agents

### Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MarketPulseRequest                          │
│  user_id, selected_indices, include_portfolio, include_watchlist    │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Phase 1: MarketIntelligenceAgent                 │
├─────────────────────────────────────────────────────────────────────┤
│ INPUT:                                                              │
│   - selected_indices, time_window_hours, max_articles               │
│                                                                     │
│ OUTPUT:                                                             │
│   - market_phase        ─────────────────────────────────────┐      │
│   - indices_data        ─────────────────────────────────────┤      │
│   - market_outlook      ─────────────────────────────────────┤      │
│   - news_items          ─────────────────────────────────────┤      │
│   - preliminary_themes  ─────────────────────────────────────┤      │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Phase 2: PortfolioInsightAgent                   │
├─────────────────────────────────────────────────────────────────────┤
│ INPUT (from Phase 1):                                               │
│   - news_items, indices_data, preliminary_themes                    │
│                                                                     │
│ INPUT (from Request):                                               │
│   - user_id, include_watchlist, include_portfolio                   │
│                                                                     │
│ OUTPUT:                                                             │
│   - news_with_impacts   ─────────────────────────────────────┐      │
│   - portfolio_impact    ─────────────────────────────────────┤      │
│   - refined_themes      ─────────────────────────────────────┤      │
│   - watchlist_alerts    ─────────────────────────────────────┤      │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Phase 3: SummaryGenerationAgent                   │
├─────────────────────────────────────────────────────────────────────┤
│ INPUT (from Phase 1):                                               │
│   - market_outlook, market_phase, indices_data                      │
│                                                                     │
│ INPUT (from Phase 2):                                               │
│   - news_with_impacts, portfolio_impact, refined_themes             │
│                                                                     │
│ OUTPUT:                                                             │
│   - market_summary_bullets                                          │
│   - trending_now_section                                            │
│   - executive_summary                                               │
│   - key_takeaways                                                   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       MarketPulseResponse                           │
│  (Assembled by Orchestrator from all agent outputs)                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Error Handling & Degraded Mode

### Agent Failure Handling

The orchestrator implements graceful degradation:

| Failed Agent | Impact | Behavior |
|--------------|--------|----------|
| `MarketIntelligenceAgent` | **Critical** | Cannot proceed, return fallback response |
| `PortfolioInsightAgent` | Non-critical | Continue with defaults, set `degraded_mode=True` |
| `SummaryGenerationAgent` | Non-critical | Use basic summary, set `degraded_mode=True` |

### Degraded Response Example

```python
# When market intelligence fails:
MarketPulseResponse(
    market_phase="mid",  # Safe default
    all_news=[],
    themed_news=[],
    portfolio_impact_summary="Market data temporarily unavailable",
    degraded_mode=True,
    warnings=["Market data unavailable", "Market intelligence failed, cannot proceed"]
)
```

### Retry Logic

```python
# BaseAgent.execute_with_retry() handles:
# 1. Timeout errors → Log, retry with backoff
# 2. Reasoning errors → Log, retry
# 3. General exceptions → Log, retry

# Backoff formula:
delay = config.retry_delay_seconds * (attempt + 1)
# Attempt 1: 1.0s delay
# Attempt 2: 2.0s delay
```

---

## Adding a New Agent

### Step 1: Define Input/Output Schemas

In `app/models/agent_schemas.py`:

```python
class MyNewAgentInput(BaseModel):
    """Input schema for My New Agent."""
    required_field: str = Field(..., description="Description")
    optional_field: int = Field(default=10, ge=1, le=100)

class MyNewAgentOutput(BaseModel):
    """Output schema for My New Agent."""
    result: str = Field(..., description="The result")
    confidence: float = Field(..., ge=0.0, le=1.0)
```

### Step 2: Create Agent Class

In `app/agents/my_new_agent.py`:

```python
from app.agents.base import BaseAgent, AgentConfig, AgentExecutionContext
from app.models.agent_schemas import MyNewAgentInput, MyNewAgentOutput

class MyNewAgent(BaseAgent[MyNewAgentInput, MyNewAgentOutput]):
    """Description of what this agent does."""
    
    input_schema = MyNewAgentInput
    output_schema = MyNewAgentOutput
    
    def __init__(self):
        config = AgentConfig(
            name="my_new_agent",
            description="What this agent does",
            model_name="gemini-2.0-flash-exp",
            temperature=0.1,
            max_output_tokens=4096,
            timeout_seconds=30,
            retry_attempts=2,
        )
        super().__init__(config)
    
    def get_system_prompt(self) -> str:
        return """You are an expert at...
        
        Your responsibilities:
        1. ...
        2. ...
        """
    
    async def execute(
        self,
        input_data: MyNewAgentInput,
        context: AgentExecutionContext,
    ) -> MyNewAgentOutput:
        self.logger.info("executing", field=input_data.required_field)
        
        # Your logic here
        result = await self._do_something(input_data)
        
        return MyNewAgentOutput(
            result=result,
            confidence=0.85
        )
    
    async def _do_something(self, input_data: MyNewAgentInput) -> str:
        # Helper method
        pass
```

### Step 3: Add Tools (Optional)

If your agent needs to call external functions:

```python
from app.tools.my_tools import get_my_tools, get_my_tool_handlers

class MyNewAgent(BaseAgent[MyNewAgentInput, MyNewAgentOutput]):
    def __init__(self):
        # ... config ...
        super().__init__(config)
        
        # Register tool handlers
        for name, handler in get_my_tool_handlers().items():
            self.register_tool_handler(name, handler)
    
    def get_tools(self):
        return get_my_tools()
```

### Step 4: Export from `__init__.py`

```python
from app.agents.my_new_agent import MyNewAgent

__all__ = [
    # ... existing exports ...
    "MyNewAgent",
]
```

### Step 5: Integrate with Orchestrator (if needed)

Add a new phase in `orchestrator.py`:

```python
# In __init__
self.my_new_agent = MyNewAgent()

# Add new phase
async def _execute_phase_n(self, ...) -> AgentExecutionResult:
    my_input = MyNewAgentInput(...)
    return await self.my_new_agent.execute_with_retry(my_input, context)
```

---

## Best Practices

### 1. Keep Agents Focused

Each agent should have a **single responsibility**. If an agent is doing too many things, split it.

### 2. Use Type-Safe Schemas

Always define Pydantic schemas for input/output. This catches errors early:

```python
# Good: Type-safe
class AgentOutput(BaseModel):
    sentiment: Literal["bullish", "bearish", "neutral"]

# Bad: Stringly-typed
output = {"sentiment": sentiment}  # No validation
```

### 3. Log Meaningful Events

Use structured logging for debugging:

```python
self.logger.info(
    "analyzing_news",
    news_count=len(news_items),
    user_id=context.user_id,
    request_id=context.request_id,
)
```

### 4. Handle Missing Data Gracefully

Always check for `None` and provide defaults:

```python
market_outlook = phase1_result.output.market_outlook if phase1_result.output else None
indices_data = phase1_result.output.indices_data if phase1_result.output else {}
```

### 5. Set Appropriate Timeouts

Base timeout on expected operation time:

```python
# Network calls: Higher timeout
timeout_seconds=settings.MARKET_DATA_AGENT_TIMEOUT + 10

# Pure computation: Lower timeout
timeout_seconds=15
```

### 6. Temperature Guidelines

| Use Case | Temperature |
|----------|-------------|
| Data extraction | 0.0 - 0.1 |
| Analysis | 0.1 - 0.2 |
| Creative writing | 0.3 - 0.5 |

### 7. Test Agents in Isolation

Each agent can be tested independently:

```python
agent = MarketIntelligenceAgent()
result = await agent.execute_with_retry(
    MarketIntelligenceAgentInput(selected_indices=["NIFTY 50"]),
    AgentExecutionContext(request_id="test-123", user_id="test-user")
)
assert result.status == "success"
```

---

## Quick Reference

### Agent Responsibilities Cheat Sheet

| Agent | Gets | Produces | Key Feature |
|-------|------|----------|-------------|
| `MarketIntelligenceAgent` | Indices, time range | Market data, news, themes | Determines market phase |
| `PortfolioInsightAgent` | User ID, news | Portfolio impact, alerts | Causal chain analysis |
| `SummaryGenerationAgent` | All prior data | Human-readable summaries | Causal language |
| `OrchestratorAgent` | Request | Final response | Phase coordination |

### Common Patterns

```python
# Get agent result
result = await agent.execute_with_retry(input, context)
if result.status == "success":
    output = result.output
else:
    handle_failure(result.error)

# Use LLM with tools
response = await self.generate_with_model(prompt, use_tools=True)

# Parse structured output
parsed = self.parse_response(response_text, OutputSchema)
```

---

## Questions?

If you have questions about the agent system:
1. Check the type definitions in `app/models/agent_schemas.py`
2. Look at the prompts in `app/prompts/` for agent behavior
3. Check the tools in `app/tools/` for external capabilities
4. Review the services in `app/services/` for data fetching
