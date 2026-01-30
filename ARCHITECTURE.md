# Market Intelligence System - Clean Architecture

> **Last Updated:** Project cleanup completed - removed real-time agent code, keeping only background processing architecture.

## Overview

This system provides real-time market intelligence through a background processing architecture using Celery tasks and MongoDB storage. The API serves pre-computed snapshots for sub-200ms response times.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Request                               │
│                    GET /api/v1/market-summary                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MongoDB                                    │
│  ┌──────────────────┐  ┌────────────────┐  ┌─────────────────┐ │
│  │  market_snapshots │  │ news_articles  │  │indices_timeseries│ │
│  │  (TTL: 15 min)    │  │ (90 day retain)│  │ (90 day TTL)    │ │
│  └──────────────────┘  └────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Celery Tasks (Background)                     │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │fetch_news (15m) │  │gen_snapshot (30m)│  │fetch_indices   │ │
│  │                 │  │                  │  │(5m market hrs) │ │
│  └────────┬────────┘  └────────┬─────────┘  └───────┬────────┘ │
│           │                    │                     │          │
│           ▼                    ▼                     ▼          │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │NewsProcessing   │  │SnapshotGeneration│  │IndicesCollection│
│  │Agent (Gemini)   │  │Agent (Gemini)    │  │Agent           │ │
│  └─────────────────┘  └──────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     Data Sources (CMOTS API)                     │
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────────┐│
│  │ News APIs       │  │ World Indices│  │ Company News        ││
│  │ (6 categories)  │  │              │  │                     ││
│  └─────────────────┘  └──────────────┘  └─────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 3-Agent Architecture

### 1. NewsProcessingAgent
**Location:** `app/agents/news_processing_agent.py`

**Responsibilities:**
- AI-powered sentiment analysis (bullish/bearish/neutral)
- Entity extraction (stocks, sectors, companies)
- Summary generation
- Impact analysis with causal chains

**Called by:** `fetch_and_process_news` Celery task (every 15 minutes)

**AI Integration:** Uses Gemini via `NewsProcessorService`

### 2. SnapshotGenerationAgent
**Location:** `app/agents/snapshot_agent.py`

**Responsibilities:**
- Market outlook generation (pre/post market only)
- Summary bullets with mandatory causal language
- Executive summary generation
- Trending news selection (mid-market)

**Called by:** `generate_market_snapshot` Celery task (every 30 minutes)

**AI Integration:** Uses Gemini via `SnapshotGeneratorService`

### 3. IndicesCollectionAgent
**Location:** `app/agents/indices_agent.py`

**Responsibilities:**
- Fetch world indices from CMOTS API
- Store historical data in MongoDB
- Market hours awareness (9:15 AM - 3:30 PM IST)
- Data normalization

**Called by:** `fetch_indices_data` Celery task (every 5 minutes during market hours)

## Celery Task Schedule

| Task | Interval | Agent | Purpose |
|------|----------|-------|---------|
| `fetch_and_process_news` | 15 min | NewsProcessingAgent | Fetch news, dedupe, AI analyze |
| `generate_market_snapshot` | 30 min | SnapshotGenerationAgent | Generate aggregated snapshot |
| `fetch_indices_data` | 5 min | IndicesCollectionAgent | Store historical indices |
| `cleanup_old_data` | Daily 2 AM | - | Archive old data |

## API Endpoints

### Primary Endpoint
```
GET /api/v1/market-summary
```
- Reads ONLY from MongoDB snapshots (no real-time AI processing)
- Response time target: < 200ms
- Falls back to stale data if no fresh snapshot
- Triggers async snapshot generation if none exists

### Supporting Endpoints
```
POST /api/v1/snapshot/generate    # Manual snapshot trigger
GET  /api/v1/db/stats             # Database statistics
POST /api/v1/data/populate        # Manual data population
```

## MongoDB Collections

### 1. market_snapshots
- **Purpose:** Pre-computed market views
- **TTL:** 15 minutes (configurable)
- **Indexes:** `generated_at`, `market_phase`, `snapshot_id`, `expires_at`

### 2. news_articles
- **Purpose:** Processed news with AI analysis
- **Retention:** 90 days
- **Indexes:** `news_id`, `published_at`, `sentiment`, `mentioned_stocks`

### 3. indices_timeseries
- **Purpose:** Historical indices data
- **TTL:** 90 days
- **Indexes:** `ticker + timestamp`, `timestamp`

## Data Flow

### News Processing Flow
```
CMOTS API → fetch_and_process_news → NewsProcessingAgent (AI) → MongoDB
```

### Snapshot Generation Flow
```
MongoDB (news + indices) → generate_market_snapshot → SnapshotGenerationAgent (AI) → MongoDB
```

### API Response Flow
```
API Request → MongoDB (snapshot) → Response (< 200ms)
```

## Configuration

Key settings in `app/config.py`:

```python
# Task Schedules
NEWS_FETCH_INTERVAL = 900        # 15 minutes
SNAPSHOT_GENERATION_INTERVAL = 1800  # 30 minutes
INDICES_FETCH_INTERVAL = 300     # 5 minutes

# Data Retention
SNAPSHOT_TTL_SECONDS = 86400     # 1 day
NEWS_RETENTION_DAYS = 90
INDICES_RETENTION_DAYS = 90

# Market Hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30
```

## Docker Services

```yaml
services:
  market-pulse-api      # FastAPI application
  celery-worker         # Background task processor
  celery-beat           # Task scheduler
  mongodb               # Data storage
  redis                 # Celery broker + caching
  flower                # Celery monitoring (optional)
```

## Running the System

```bash
# Start core services
docker-compose up -d

# Start with monitoring
docker-compose --profile monitoring up -d

# Manual task triggers
curl -X POST "http://localhost:8000/api/v1/data/populate"
curl "http://localhost:8000/api/v1/market-summary"
```

## Key Design Principles

1. **Separation of Concerns:** API serves cached data, background tasks handle heavy processing
2. **Graceful Degradation:** Rule-based fallbacks when AI fails
3. **Deduplication:** News articles deduplicated by `news_id` (sno from API)
4. **TTL Management:** MongoDB handles automatic expiration
5. **Market Awareness:** Indices collection respects market hours
