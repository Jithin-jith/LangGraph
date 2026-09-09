# Module 05: Advanced State Management & Custom Reducers

This module explores advanced state management patterns in LangGraph, focusing on strict type safety with Pydantic V2 schemas and custom state reducer aggregation functions.

---

## 🔑 Key Concepts Covered

1. **Pydantic V2 State Schemas (`BaseModel`)**:
   - Passing a Pydantic `BaseModel` directly to `StateGraph(MyPydanticState)` replaces standard `TypedDict`.
   - Enforces strict runtime field validation, automatic type conversion, and self-documenting field metadata using `Field(description=..., default_factory=...)`.
   - Prevents bugs caused by invalid types or missing required state attributes.

2. **Custom Reducer Functions (`typing.Annotated`)**:
   - By default, LangGraph nodes overwrite existing state keys unless a reducer is defined.
   - Built-in reducers (such as `operator.add` or `add_messages`) append state updates unconditionally.
   - Custom reducers function with the signature `def my_reducer(existing: T, new: T) -> T` allow custom state mutation logic:
     - **List Deduplication**: Merging new elements while preserving sequence order and removing duplicate entries.
     - **Sliding Window Capping**: Appending events while retaining only the last N items to prevent context window bloat.
     - **Custom Dictionary Merging**: Selective key updates and conflict resolution.

---

## 📁 Lesson Scripts Breakdown

### 1. [`01_pydantic_state_schema.py`](01_pydantic_state_schema.py): Type-Safe Pydantic V2 Graph State
- **Goal**: Build an AI-powered research pipeline (`query_generator` -> `researcher` -> `summarizer`) with a strictly validated Pydantic V2 `ResearchReportState`.
- **Workflow**:
  - **Schema (`ResearchReportState`)**: Declares `topic` (str), `query_list` (List[str]), `raw_findings` (List[str]), `summary` (Optional[str]), and `word_count` (int).
  - **Node 1 (`query_generator_node`)**: Uses Gemini (`gemini-2.5-flash`) to generate research queries for the target topic.
  - **Node 2 (`researcher_node`)**: Collects research facts for generated queries.
  - **Node 3 (`summarizer_node`)**: Synthesizes notes into an executive summary and calculates total word count.
  - **Validation**: State updates returned by each node are auto-validated against Pydantic schema constraints.

### 2. [`02_custom_reducers.py`](02_custom_reducers.py): Custom State Reducer Functions
- **Goal**: Implement custom aggregation logic for list deduplication and sliding-window event log capping across sequential graph nodes (`alpha` -> `beta` -> `gamma`).
- **Workflow**:
  - **Custom Reducer 1 (`deduplicate_reducer`)**: Appends items to `tags` while eliminating duplicates (`"ai"` and `"python"` emitted across multiple nodes appear only once).
  - **Custom Reducer 2 (`sliding_window_reducer`)**: Appends event logs to `recent_events` while retaining only the last 3 items (`Event 1` and `Event 2` are pruned when `Event 5` arrives).
  - **Schema (`AdvancedReducerState`)**: Binds reducers via `typing.Annotated[list[str], deduplicate_reducer]` and `typing.Annotated[list[str], sliding_window_reducer]`.

---

## 🚀 Running the Lessons

Run each script from the terminal to observe Pydantic validation and custom state reducer operations:

```bash
# Test Pydantic V2 state schema research pipeline:
python Module_05_Advanced_State/01_pydantic_state_schema.py

# Test custom list deduplication & sliding window reducers:
python Module_05_Advanced_State/02_custom_reducers.py
```
