# Module 08: Dynamic Workflows & Map-Reduce / Parallel Execution

This module covers concurrent node execution strategies in LangGraph, from static fan-out/fan-in branching to dynamic runtime map-reduce task generation using the `Send` API.

---

## 🔑 Key Concepts Covered

1. **Parallel Fan-Out & Fan-In Execution**:
   - Connecting `START -> [Node A, Node B, Node C]` executes all 3 branches concurrently.
   - Connecting `[Node A, Node B, Node C] -> Node Aggregator` pauses execution until all parallel branches complete, consolidating returned results via accumulative list reducers (`Annotated[list, add]`).

2. **Dynamic Map-Reduce via `Send` API**:
   - `Send("target_node", {"key": item})` dynamically instantiates graph node tasks at runtime based on an input list length.
   - Enables processing variable-length collections (e.g. 3 or 30 topics) in parallel without pre-declaring fixed static node edges.

---

## 📁 Lesson Scripts Breakdown

### 1. [`01_parallel_branches.py`](01_parallel_branches.py): Static Fan-Out & Fan-In Execution
- **Goal**: Analyze a target company concurrently across 3 specialized dimensions (financial, tech stack, competitors) and aggregate results into a dossier.
- **Workflow**:
  - **Parallel Fan-Out**: `START` triggers `financial_analyst`, `tech_analyst`, and `competitor_analyst` simultaneously.
  - **Fan-In Aggregation**: All 3 analyst nodes connect to `aggregator_node` which combines `branch_results` into `consolidated_report`.

### 2. [`02_map_reduce_send.py`](02_map_reduce_send.py): Dynamic Map-Reduce with `Send` API
- **Goal**: Dynamically map over a list of technology topics, summarize each in parallel, and reduce findings into an executive tech digest.
- **Workflow**:
  - **Map Router (`spawn_map_tasks`)**: Reads `subjects` list and generates `Send("summarize_subject_worker", {"subject": topic})` objects.
  - **Worker Tasks (`summarize_subject_worker`)**: Summarizes an individual topic and appends the result to `summaries`.
  - **Reduce Node (`reduce_results_node`)**: Compiles all summaries into `final_digest`.

---

## 🚀 Running the Lessons

Run each script from the terminal to observe static parallel branches and dynamic `Send` API map-reduce execution:

```bash
# Test static parallel fan-out and fan-in graph execution:
python Module_08_Parallel_and_Dynamic/01_parallel_branches.py

# Test dynamic map-reduce with Send API:
python Module_08_Parallel_and_Dynamic/02_map_reduce_send.py
```
