# Module 06: Subgraphs & Modular Architecture

Subgraphs allow developers to modularize complex graph architectures by breaking them down into encapsulated, reusable child graphs.

---

## 🔑 Key Concepts Covered

1. **Subgraph Composition**:
   - A child `StateGraph` compiled with `.compile()` can be added directly as a node inside a parent graph via `parent_builder.add_node("subgraph_name", compiled_subgraph)`.
   - Allows complex workflows (e.g., critique-refinement loops or multi-step code auditing) to be packaged into clean, reusable components.

2. **Direct Embedding vs. State Mapping Wrappers**:
   - **Direct Embedding**: When parent state and child state share identical key names (e.g., `draft`, `critique`), state flows seamlessly between parent and child without manual mapping.
   - **State Mapping Wrappers**: When parent state (`EnterprisePipelineStates`) differs from child state (`CodeReviewState`), wrap the subgraph inside a translation node function that converts `ParentState -> SubgraphState` before invocation and `SubgraphState -> ParentState` after invocation.

---

## 📁 Lesson Scripts Breakdown

### 1. [`01_nested_subgraph.py`](01_nested_subgraph.py): Nested Subgraph Composition
- **Goal**: Build an AI pitch generation pipeline that embeds a child essay-refinement subgraph as a single node in a parent graph.
- **Workflow**:
  - **Child Subgraph (`refinement_subgraph`)**: Takes `draft` -> runs `generate_critique` -> runs `revise_draft` -> outputs `revised_draft`.
  - **Parent Graph**: Writes initial pitch (`initial_pitch`) -> delegates to `refinement_subgraph` -> formats final delivery pitch (`finalize_pitch`).

### 2. [`02_state_mapping_subgraphs.py`](02_state_mapping_subgraphs.py): Parent-Child State Mapping Wrappers
- **Goal**: Audit source code using an isolated `CodeReviewState` subgraph invoked inside an enterprise pipeline using `EnterprisePipelineStates`.
- **Workflow**:
  - **Child Subgraph (`code_review_subgraph`)**: Operates on `code_snippet` -> audits `check_security` -> audits `check_performance`.
  - **Parent Wrapper Node (`run_code_review_wrapper`)**: Extracts `raw_python_code` from parent state -> maps to `code_snippet` -> invokes child subgraph -> maps security & performance results into parent `audit_report`.

---

## 🚀 Running the Lessons

Run each script from the terminal to observe nested and state-mapped subgraph execution:

```bash
# Test direct nested subgraph execution:
python Module_06_Subgraphs/01_nested_subgraph.py

# Test state mapping wrapper node for subgraphs:
python Module_06_Subgraphs/02_state_mapping_subgraphs.py
```
