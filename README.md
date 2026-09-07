# 🚀 LangGraph Masterclass: Beginner to Advanced (Gemini Edition)

Welcome to the comprehensive, hands-on **LangGraph Curriculum**! This repository contains 10 structured, progressive modules designed to take you from core graph fundamentals to advanced multi-agent systems, observability, evaluations, and production deployment.

All examples use **Google Gemini models** (`gemini-2.5-flash`) via `langchain-google-genai`.

---

## 📚 Curriculum Structure Overview

| Module | Title | Key Topics Covered |
| :--- | :--- | :--- |
| **[Module 01](Module_01_Core_Fundamentals/README.md)** | Core Fundamentals | `StateGraph`, `TypedDict` state, Nodes, Edges, `add_conditional_edges`, Reducers (`Annotated[list, add]`) |
| **[Module 02](Module_02_Chatbots_and_Tools/README.md)** | Chatbots & Tool Integration | `MessagesState`, `bind_tools`, `ToolNode`, `tools_condition`, structured tools, fallback error handling |
| **[Module 03](Module_03_Memory_and_Persistence/README.md)** | Memory & Persistence | `MemorySaver`, `SqliteSaver`, thread isolation (`thread_id`), state snapshots (`get_state`), Time Travel (`update_state`) |
| **[Module 04](Module_04_Human_in_the_Loop/README.md)** | Human-in-the-Loop (HITL) | Breakpoints (`interrupt_before`), human approval workflows, state injection/editing before resuming execution |
| **[Module 05](Module_05_Advanced_State/README.md)** | Advanced State Management | Pydantic V2 state models, custom reducer functions (deduplication, sliding window) |
| **[Module 06](Module_06_Subgraphs/README.md)** | Subgraphs & Modular Architecture | Embedding child graphs inside parent graphs, parent-child state mapping wrappers |
| **[Module 07](Module_07_Multi_Agent_Systems/README.md)** | Multi-Agent Architectures | Generator-Critic reflection loops, Supervisor pattern, dynamic worker orchestration |
| **[Module 08](Module_08_Parallel_and_Dynamic/README.md)** | Parallel & Dynamic Workflows | Fan-Out / Fan-In concurrency, dynamic task generation using `Send` API map-reduce |
| **[Module 09](Module_09_Observability_and_Streaming/README.md)** | Observability & Streaming | LangSmith telemetry & tracing, real-time output streaming (`stream_mode="updates"`), error resilience & fallbacks |
| **[Module 10](Module_10_Testing_Evaluation_Production/README.md)** | Testing, Eval & Production | Pytest unit/integration testing, LLM-as-a-Judge benchmark scoring, FastAPI REST service deployment |

---

## ⚡ Quick Start & Setup

### 1. Prerequisites
- Python 3.10+
- Google Gemini API Key (`GOOGLE_API_KEY`)

### 2. Environment Setup
```bash
# Clone or navigate to the repository
cd f:/Projects/LangGraph

# Install dependencies
pip install -r requirements.txt
```

Ensure your `.env` file in the root directory contains your Google Gemini API Key:
```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
LANGCHAIN_API_KEY=optional_langsmith_key_for_module_09
```

---

## 🏃 Running the Code Examples

Navigate to any module or execute directly from the root:

```bash
# Module 01: Core Graph Fundamentals
python Module_01_Core_Fundamentals/01_basic_graph.py
python Module_01_Core_Fundamentals/02_conditional_router.py
python Module_01_Core_Fundamentals/03_state_reducers.py

# Module 02: Chatbots & Tool Integration
python Module_02_Chatbots_and_Tools/01_basic_chatbot.py
python Module_02_Chatbots_and_Tools/02_tool_calling_agent.py
python Module_02_Chatbots_and_Tools/03_custom_tools_and_fallbacks.py

# Module 03: Memory & Persistence
python Module_03_Memory_and_Persistence/01_in_memory_checkpointer.py
python Module_03_Memory_and_Persistence/02_sqlite_persistence.py
python Module_03_Memory_and_Persistence/03_time_travel_and_replay.py
python Module_03_Memory_and_Persistence/04_multi_user_sqlite_chatbot.py

# Module 04: Human-in-the-Loop
python Module_04_Human_in_the_Loop/01_interrupt_approval.py
python Module_04_Human_in_the_Loop/02_state_modification_hitl.py

# Module 05: Advanced State
python Module_05_Advanced_State/01_pydantic_state_schema.py
python Module_05_Advanced_State/02_custom_reducers.py

# Module 06: Subgraphs
python Module_06_Subgraphs/01_nested_subgraph.py
python Module_06_Subgraphs/02_state_mapping_subgraphs.py

# Module 07: Multi-Agent Systems
python Module_07_Multi_Agent_Systems/01_reflection_agent.py
python Module_07_Multi_Agent_Systems/02_supervisor_agent.py

# Module 08: Parallel & Dynamic Workflows
python Module_08_Parallel_and_Dynamic/01_parallel_branches.py
python Module_08_Parallel_and_Dynamic/02_map_reduce_send.py

# Module 09: Observability & Streaming
python Module_09_Observability_and_Streaming/01_langsmith_tracing.py
python Module_09_Observability_and_Streaming/02_token_and_event_streaming.py
python Module_09_Observability_and_Streaming/03_resilience_and_fallbacks.py

# Module 10: Testing, Evaluation & Production
pytest Module_10_Testing_Evaluation_Production/01_unit_testing.py
python Module_10_Testing_Evaluation_Production/02_llm_as_judge_evaluation.py
python Module_10_Testing_Evaluation_Production/03_production_service.py
```

---

## 🛠️ Tech Stack & Key Libraries Used
- **[LangGraph](https://github.com/langchain-ai/langgraph)** `v1.2+`: Stateful multi-actor orchestration.
- **[langchain-google-genai](https://github.com/langchain-ai/langchain-google)**: Integration with Google Gemini LLMs (`gemini-2.5-flash`).
- **[Pydantic V2](https://docs.pydantic.dev/)**: Data validation & schema definition.
- **[FastAPI](https://fastapi.tiangolo.com/)**: REST API production deployment.
- **[Pytest](https://docs.pytest.org/)**: Automated unit testing.
- **[LangSmith](https://smith.langchain.com/)**: Observability, tracing & telemetry.