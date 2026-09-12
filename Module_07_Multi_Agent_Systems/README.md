# Module 07: Multi-Agent Architectures

Multi-agent systems decompose complex tasks into coordinated networks of specialized AI agents working together towards a shared goal.

---

## 🔑 Key Concepts Covered

1. **Reflection & Self-Correction Pattern**:
   - A **Generator** node creates initial draft outputs, while a **Critic** node reviews draft outputs against editorial/quality criteria.
   - A conditional edge loops back to the generator node until the critic approves output or max iteration limits are reached.

2. **Supervisor / Orchestrator Pattern**:
   - A central **Supervisor** node examines global state and dynamically decides which specialized worker node (`researcher`, `coder`, `FINISH`) should run next.
   - Worker nodes execute autonomous sub-tasks and return control back to the supervisor upon completion.

---

## 📁 Lesson Scripts Breakdown

### 1. [`01_reflection_agent.py`](01_reflection_agent.py): Generator-Critic Reflection Loop
- **Goal**: Implement a self-correcting writing pipeline using a Generator agent and a Critic agent.
- **Workflow**:
  - **Generator Node**: Drafts text overview on iteration 1; revises text based on Critic feedback on subsequent iterations.
  - **Critic Node**: Evaluates draft quality and provides actionable improvements or returns `APPROVED`.
  - **Router (`should_continue`)**: Checks if `APPROVED` or max iterations reached (`end`) vs. requesting revisions (`generator`).

### 2. [`02_supervisor_agent.py`](02_supervisor_agent.py): Supervisor Orchestrator Architecture
- **Goal**: Build a hub-and-spoke multi-agent system that coordinates technical research and code generation.
- **Workflow**:
  - **Supervisor Node**: Inspects state fields (`research_notes`, `code_output`) to decide whether to invoke `researcher`, `coder`, or `FINISH`.
  - **Researcher Worker**: Collects technical requirements for the user task.
  - **Coder Worker**: Generates Python code implementations from technical research notes.

---

## 🚀 Running the Lessons

Run each script from the terminal to observe multi-agent reflection and supervisor orchestration:

```bash
# Test generator-critic reflection loop:
python Module_07_Multi_Agent_Systems/01_reflection_agent.py

# Test supervisor multi-agent orchestrator:
python Module_07_Multi_Agent_Systems/02_supervisor_agent.py
```
