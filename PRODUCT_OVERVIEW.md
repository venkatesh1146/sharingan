# Market Pulse Engine — Product Overview

**For product managers and stakeholder presentations**

---

## One-line pitch

**Market Pulse** is an AI-powered market intelligence system that turns news and indices into actionable snapshots—delivered in under 200 ms so apps and dashboards stay fast and up to date.

---

## The problem we solve

- **Information overload:** Teams drown in raw news and indices; they need a single, interpreted view.
- **Slow or manual analysis:** Real-time AI on every request is slow and expensive; manual summaries don’t scale.
- **Inconsistent quality:** Ad-hoc summaries lack structure, causality, and traceability to sources.

---

## Our solution

We run **background AI agents** that continuously fetch news, process indices, and generate **pre-computed market snapshots**. The API only reads these snapshots—no AI on request—so responses are **fast (< 200 ms)**, **consistent**, and **cost-efficient**.

| Before | After |
|--------|--------|
| Raw news feeds + manual reading | One structured snapshot: outlook, summary bullets, executive summary |
| Real-time AI per request (slow, costly) | Pre-computed snapshots served from storage |
| Unclear why the market moved | Causal language and news-backed reasoning |

---

## Key product capabilities

### 1. **Market outlook**
- Sentiment: **bullish / bearish / neutral** with confidence score  
- Short reasoning tied to Nifty and key drivers  
- Aligned with pre-market, live, or post-market phase  

### 2. **Structured market summary**
- Bullet-style summary points with **mandatory causal language** (why, not just what)  
- Each point linked to **supporting news** for auditability  
- Sentiment and confidence per point  

### 3. **Executive summary**
- One short paragraph for leadership and dashboards  
- Phase-aware (pre / mid / post market)  

### 4. **Trending and themed news** (mid-market)
- Curated “trending now” and themed news when relevant  
- Surfaces what matters without scrolling raw feeds  

### 5. **News intelligence**
- **Per-article:** sentiment, entities (stocks, sectors, companies), summary, impact/causal chain  
- **Deduplication** so the same story isn’t counted twice  
- **90-day** searchable history of processed news  

### 6. **Indices context**
- World indices (including Nifty) ingested on a schedule  
- **Market-hours aware** (e.g. 9:15–15:30 IST); historical series retained for 90 days  
- Used to ground outlook and summary in actual index moves  

---

## Who it’s for

- **Internal dashboards:** Pre-market briefs, live monitors, end-of-day wrap-ups  
- **Trading / research teams:** Quick view of sentiment and drivers without opening 10 tabs  
- **Product/API consumers:** Any app that needs a stable, fast “market summary” API  
- **Compliance / audit:** Trace summary points back to specific news and timestamps  

---

## How it works (simplified)

```
Data sources (e.g. CMOTS: news + indices)
        ↓
Background tasks (on a schedule)
        ↓
3 AI agents: News processing → Indices collection → Snapshot generation
        ↓
Stored snapshots (with TTL) + processed news (90-day retention)
        ↓
API: GET market-summary → returns latest snapshot (< 200 ms)
```

- **News:** Fetched and processed every few minutes; each article gets sentiment, entities, summary, impact.  
- **Snapshots:** Generated on a schedule (e.g. every 5 minutes), using latest news + indices + previous snapshot for continuity.  
- **API:** Serves the latest snapshot only; if none exists, can trigger generation asynchronously and still return the last available snapshot (graceful degradation).  

No need to explain Celery/MongoDB in the room—focus on “scheduled AI pipelines” and “pre-built snapshots.”

---

## Key metrics and targets

| Metric | Target | Why it matters |
|--------|--------|----------------|
| **API response time** | < 200 ms | Dashboards and apps feel instant |
| **Snapshot freshness** | Every 5 min (configurable) | Balances freshness vs. cost and load |
| **News processing** | Every 3 min | New stories reflected in next snapshot |
| **Data retention** | Snapshots: TTL-based; News: 90 days | Enough history for review and audit |
| **Degradation** | Rule-based fallbacks if AI fails | Service stays up; quality may be reduced |

These are the numbers PM and stakeholders can promise and monitor.

---

## What the API delivers

**Primary endpoint:** `GET /api/v1/market-summary`

Returns a single JSON payload containing:

- **generated_at** — When the snapshot was produced  
- **market_phase** — pre / live / post  
- **market_outlook** — Sentiment, confidence, reasoning, Nifty change %, key drivers  
- **indices_data** — Current index levels and changes  
- **market_summary** — Bullet points with causal language and supporting news IDs  
- **executive_summary** — One-paragraph summary  
- **trending_now** / **themed_news** — When applicable  
- **all_news_ids** — Full list of news used in the snapshot (for drill-down or audit)  
- **degraded_mode** / **warnings** — Transparency when fallbacks are used  

Supporting endpoints: manual snapshot trigger, DB stats, data population, health, agent status—useful for ops and support.

---

## Design principles (for PM storytelling)

1. **Speed first:** API serves cache; heavy work happens in the background.  
2. **Graceful degradation:** If AI fails, we still return a snapshot with rule-based content where possible.  
3. **Traceability:** Every summary point can be tied back to specific news and timestamps.  
4. **Continuity:** New snapshots build on the previous phase snapshot so the narrative doesn’t reset every run.  
5. **Controlled cost:** No per-request AI; processing is batched and scheduled.

---

## Slide-friendly summary bullets

- **Product:** AI-powered market intelligence via pre-computed snapshots.  
- **Differentiator:** Sub-200 ms API; causal, news-backed summaries; phase-aware (pre/live/post).  
- **Users:** Dashboards, trading/research teams, any app needing a market summary API.  
- **Delivery:** REST API; primary product is `GET /api/v1/market-summary`.  
- **Quality:** Structured output, sentiment, confidence, and fallbacks when AI fails.  
- **Performance:** < 200 ms response; snapshots every 5 min; news every 3 min.  

---

## Optional: future product angles

- **Themes and watchlists:** Deeper themed news and watchlist/portfolio impact.  
- **Alerts:** Notifications when sentiment or key drivers cross thresholds.  
- **Custom schedules:** Different snapshot cadences per client or use case.  
- **Multi-region / multi-market:** More indices and news sources with the same snapshot model.  

Use these for roadmap or “what’s next” slides.

---

*Last updated from ARCHITECTURE.md, v2.md, and README.md. For technical details, see ARCHITECTURE.md and README.md.*
