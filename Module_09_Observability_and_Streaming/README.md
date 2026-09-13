# Module 09: Observability, Tracing & Streaming

Observability, real-time event streaming, and fault tolerance are essential requirements for operating production-grade LangGraph state machines.

---

## 🔑 Key Concepts Covered

1. **LangSmith Telemetry & Tracing**:
   - Setting environment variables (`LANGCHAIN_TRACING_V2="true"`, `LANGCHAIN_PROJECT`) streams background telemetry, latency metrics, token consumption breakdowns, and node input/output traces to LangSmith.

2. **Graph Streaming Modes (`app.stream`)**:
   - **`stream_mode="updates"`**: Yields dictionaries containing state changes returned by individual nodes as soon as each node finishes execution.
   - **`stream_mode="values"`**: Yields complete cumulative graph state snapshots after each step in the pipeline.

3. **Fault Tolerance & Fallback Routing**:
   - Wrapping node execution in `try-except` blocks and utilizing conditional routing based on failure flags (`fallback_used=True`) enables automatic recovery using secondary fallback LLMs (`gemini-1.5-flash`) without crashing runtime execution.

---

## 📁 Lesson Scripts Breakdown

### 1. [`01_langsmith_tracing.py`](01_langsmith_tracing.py): LangSmith Telemetry & Graph Tracing
- **Goal**: Enable non-intrusive LangSmith tracing to monitor node execution times, token costs, and LLM call latency.
- **Workflow**:
  - Configures `LANGCHAIN_TRACING_V2` environment settings.
  - Runs a two-step analysis and synthesis graph (`node_analysis` -> `node_synthesis`).
  - Viewable live at [smith.langchain.com](https://smith.langchain.com).

### 2. [`02_token_and_event_streaming.py`](02_token_and_event_streaming.py): Real-Time Graph Output Streaming
- **Goal**: Compare `stream_mode="updates"` vs `stream_mode="values"` for real-time user feedback.
- **Workflow**:
  - Runs a 2-node research and summarization graph (`researcher_node` -> `summarizer_node`).
  - **`updates`**: Prints incremental node updates as each node finishes.
  - **`values`**: Prints complete cumulative state arrays after each step.

### 3. [`03_resilience_and_fallbacks.py`](03_resilience_and_fallbacks.py): Fault Tolerance & Fallback Routing
- **Goal**: Build an error-resilient workflow that catches primary LLM failures and dynamically routes to a backup fallback LLM.
- **Workflow**:
  - **Primary Node (`primary_node`)**: Attempts execution with `gemini-2.5-flash`. Catches errors and sets `fallback_used=True`.
  - **Router (`route_resilience`)**: Routes to `fallback_recovery` node if errors occurred; otherwise routes to `standard_finish`.
  - **Fallback Recovery (`fallback_recovery_node`)**: Invokes backup model (`gemini-1.5-flash`) to ensure high availability.

---

## 🚀 Running the Lessons

Run each script from the terminal to experience observability, streaming, and fallback resilience:

```bash
# Test LangSmith telemetry tracing:
python Module_09_Observability_and_Streaming/01_langsmith_tracing.py

# Test real-time graph output streaming modes:
python Module_09_Observability_and_Streaming/02_token_and_event_streaming.py

# Test error handling and fallback model recovery:
python Module_09_Observability_and_Streaming/03_resilience_and_fallbacks.py
```
