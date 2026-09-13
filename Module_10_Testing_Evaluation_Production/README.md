# Module 10: Testing, Evaluation & Production Deployment

This module covers testing methodologies, automated evaluation frameworks, and production deployment patterns for LangGraph state machines.

---

## 🔑 Key Concepts Covered

1. **Unit & Integration Testing**:
   - Isolating graph node functions for deterministic `pytest` unit testing without graph overhead.
   - Testing end-to-end state transitions across compiled `StateGraph` instances.

2. **LLM-as-a-Judge Evaluation**:
   - Automated benchmark evaluation of agent responses using structured Pydantic schemas (`EvalScore`).
   - Evaluates relevance, completeness, and provides scoring reasoning.

3. **Production FastAPI Service**:
   - Encapsulating compiled LangGraph state machines inside high-performance FastAPI REST APIs.
   - Per-thread state isolation using `MemorySaver` checkpointer and configurable `thread_id` parameters.
   - Health check endpoints (`/health`) for container orchestration probes.

---

## 📁 Lesson Scripts Breakdown

### 1. [`01_unit_testing.py`](01_unit_testing.py): Unit & Integration Testing Graphs
- **Goal**: Write unit tests for individual node logic (`clean_text_node`, `count_chars_node`) and integration tests for compiled graph flow.
- **Workflow**:
  - Node functions accept plain dictionary inputs for fast unit testing.
  - `test_full_graph_integration` verifies end-to-end state transformations across nodes.

### 2. [`02_llm_as_judge_evaluation.py`](02_llm_as_judge_evaluation.py): LLM-as-a-Judge Automated Benchmark
- **Goal**: Benchmark target agent outputs using a structured LLM evaluator (`gemini-2.5-flash` at temperature=0.0).
- **Workflow**:
  - Target agent answers benchmark questions.
  - Evaluator LLM assesses answers against expected key topics and outputs a structured Pydantic `EvalScore` (`relevance_score`, `completeness_score`, `reasoning`).

### 3. [`03_production_service.py`](03_production_service.py): Production FastAPI REST Deployment
- **Goal**: Deploy a conversational LangGraph agent as a production REST API microservice.
- **Workflow**:
  - Exposes `POST /chat` with `thread_id` session tracking via `MemorySaver`.
  - Exposes `GET /health` for Kubernetes / load balancer liveness checks.

---

## 🚀 Running the Lessons

Run each script from the terminal to execute unit tests, evaluation benchmarks, and REST API deployment:

```bash
# Run unit & integration test suite:
pytest Module_10_Testing_Evaluation_Production/01_unit_testing.py

# Run LLM-as-a-Judge automated evaluation benchmark:
python Module_10_Testing_Evaluation_Production/02_llm_as_judge_evaluation.py

# Start production FastAPI REST service server:
python Module_10_Testing_Evaluation_Production/03_production_service.py
```
