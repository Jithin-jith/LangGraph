# Module 01: LangGraph Core Fundamentals

This module covers the foundational building blocks of LangGraph state machine architectures using **Google Gemini models** (`gemini-2.5-flash`).

---

## 🔑 Core Concepts Covered

1. **Graph State (`TypedDict`)**:
   - In LangGraph, graph state is a shared Python dictionary schema passed to every node in the graph.
   - Each node receives the current state as input and returns a dictionary of state updates.

2. **Nodes & Edges**:
   - **Nodes**: Standard Python functions (or callables) that process the graph state and execute logic or LLM calls.
   - **Edges**: Connections defining execution control flow:
     - `add_edge(A, B)`: Deterministic transition from node A to node B.
     - `START` / `END`: Special graph entry and termination points.

3. **Conditional Routing (`add_conditional_edges`)**:
   - Router functions examine current state and return the string key of the next node to execute dynamically based on intent classification or runtime conditions.

4. **State Reducers (`Annotated[list, add]`)**:
   - By default, returning a key in a node's dictionary **overwrites** that key in the state.
   - When a key is typed with a reducer like `Annotated[list, add]`, returned list elements are **appended** to the existing list rather than overwriting it via Python's `operator.add` (`list1 + list2`).

---

## 📁 Detailed Breakdown of Lessons & Code Files

### 1. [`01_basic_graph.py`](01_basic_graph.py) — Linear StateGraph Pipeline

#### 🎯 Purpose & Functionality
This lesson demonstrates how to build a basic, sequential (linear) state machine using `StateGraph`. A topic input flows sequentially through 3 nodes: `greeting_node` -> `fact_node` -> `summarize_node`.

#### 🧠 How It Works
1. **State Definition**:
   ```python
   class GraphState(TypedDict):
       topic: str
       greeting: str
       fact: str
       final_output: str
   ```
2. **Node Functions**:
   - `greeting_node`: Reads `state['topic']`, invokes Gemini LLM to generate an enthusiastic welcome message, and returns `{"greeting": response}`.
   - `fact_node`: Reads `state['topic']`, invokes Gemini LLM to generate an interesting fun fact, and returns `{"fact": response}`.
   - `summarize_node`: Aggregates `state['greeting']` and `state['fact']` into a combined formatted output `{"final_output": combined}`.
3. **Graph Compilation**:
   - Edges connect sequentially: `START` -> `greeting_node` -> `fact_node` -> `summarize_node` -> `END`.
   - `app = workflow.compile()` converts the graph definition into an executable runnable object.

---

### 2. [`02_conditional_router.py`](02_conditional_router.py) — Dynamic Routing with Intent Classification

#### 🎯 Purpose & Functionality
This lesson demonstrates dynamic control flow in LangGraph. Instead of moving sequentially through fixed edges, an intent classification node uses Gemini to categorize a user query and route execution dynamically to a specialized expert node (`code_expert`, `math_expert`, or `general_expert`).

#### 🧠 How It Works
1. **Classifier Node (`classify_intent_node`)**:
   - Uses Gemini with `temperature=0.0` to analyze user prompt intent and categorize it into one of `['code', 'math', 'general']`.
   - Returns `{"category": category_name}`.
2. **Routing Function (`route_by_category`)**:
   - Inspects `state['category']` and returns a string key corresponding to the node name to execute next:
     - `'code'` -> `'code_expert'`
     - `'math'` -> `'math_expert'`
     - `'general'` -> `'general_expert'`
3. **Conditional Edge Registration**:
   ```python
   builder.add_conditional_edges(
       "classifier",        # Source node
       route_by_category,   # Decision function
       {                    # Mapping dictionary
           "code_expert": "code_expert",
           "math_expert": "math_expert",
           "general_expert": "general_expert",
       }
   )
   ```
4. **Worker Nodes**:
   - Each expert node executes specialized logic for its domain and returns `{"response": result_text}`, terminating at `END`.

---

### 3. [`03_state_reducers.py`](03_state_reducers.py) — State Mutation vs Value Accumulation

#### 🎯 Purpose & Functionality
This lesson explains how LangGraph handles state updates when nodes return data. It highlights the difference between **default overwriting** and **list appending using Reducers**.

#### 🧠 How It Works
1. **State with Reducer**:
   ```python
   from operator import add
   from typing import Annotated, TypedDict

   class ReducerState(TypedDict):
       task: str
       last_thought: str                        # Standard key -> OVERWRITES on each node return
       history_logs: Annotated[list[str], add]   # Reducer key -> APPENDS via operator.add (list1 + list2)
   ```
2. **Node Returns**:
   - When Node 1 returns `{"last_thought": "Thought 1", "history_logs": ["Log 1"]}`, `last_thought` becomes `"Thought 1"` and `history_logs` becomes `["Log 1"]`.
   - When Node 2 returns `{"last_thought": "Thought 2", "history_logs": ["Log 2"]}`, `last_thought` is **overwritten** to `"Thought 2"`, while `history_logs` uses `operator.add(["Log 1"], ["Log 2"])` to become `["Log 1", "Log 2"]`.
   - When Node 3 returns `{"last_thought": "Thought 3", "history_logs": ["Log 3"]}`, `history_logs` becomes `["Log 1", "Log 2", "Log 3"]`.

---

## 🚀 Running the Lessons

Make sure your `.env` file contains your `GOOGLE_API_KEY`, then run:

```bash
# Lesson 01: Linear Graph Pipeline
python Module_01_Core_Fundamentals/01_basic_graph.py

# Lesson 02: Dynamic Intent Router
python Module_01_Core_Fundamentals/02_conditional_router.py

# Lesson 03: State Reducers & Log Accumulation
python Module_01_Core_Fundamentals/03_state_reducers.py
```
