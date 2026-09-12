"""
Module 07 - Lesson 02: Multi-Agent Supervisor / Orchestrator Pattern
=====================================================================
Demonstrates building a centralized Supervisor agent that inspects workflow state
and dynamically routes tasks to specialized worker agents (`researcher`, `coder`).

Key Concepts Demonstrated:
---------------------------
1. Supervisor / Hub-and-Spoke Pattern: A central router node inspects state,
   assigns the next action to specialized worker nodes, and tracks overall progress.
2. Worker Node Autonomy: Workers (`researcher_agent`, `coder_agent`) perform specialized
   sub-tasks and route results BACK to the supervisor node upon completion.
3. State-Driven Routing: `supervisor_agent` evaluates state parameters (`research_notes`,
   `code_output`) to determine the next worker or issue a `FINISH` signal.
4. Dynamic Orchestration: Enables modular, multi-step problem solving where specialized
   LLM personas focus on distinct software lifecycle stages.
"""

import os
import sys
import warnings
import logging
from typing import TypedDict, Literal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ============================================================================
# SUPPRESS SDK / LOGGING VERBOSITY
# ============================================================================
class StderrFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr

    def write(self, msg):
        if "automatic function calling" in msg or "AFC" in msg:
            return
        self.original_stderr.write(msg)

    def flush(self):
        if hasattr(self.original_stderr, "flush"):
            self.original_stderr.flush()

sys.stderr = StderrFilter(sys.stderr)

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
warnings.showwarning = lambda *args, **kwargs: None

logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ============================================================================
# STEP 1: DEFINE MULTI-AGENT GRAPH STATE
# ============================================================================
# MultiAgentState holds shared state across the supervisor and worker nodes.
class MultiAgentState(TypedDict):
    task: str             # High-level task objective
    next_worker: str      # Worker key decided by supervisor ('researcher', 'coder', 'FINISH')
    research_notes: str   # Output produced by Researcher Agent
    code_output: str      # Output produced by Coder Agent
    final_response: str   # Consolidated final summary

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)


# ============================================================================
# STEP 2: DEFINE SPECIALIZED WORKER AGENTS
# ============================================================================

def researcher_agent(state: MultiAgentState) -> dict:
    """
    Worker Agent 1: Researcher Agent
    ---------------------------------
    Focuses exclusively on gathering technical specifications and key facts
    for the assigned task. Returns research_notes to state.
    """
    print("\n---> [Researcher Agent] Gathering technical specifications...")
    prompt = f"Research key technical requirements needed to build: '{state['task']}'. Provide 2 concise key points."
    res = llm.invoke(prompt).content.strip()
    return {"research_notes": res}


def coder_agent(state: MultiAgentState) -> dict:
    """
    Worker Agent 2: Coder Agent
    ---------------------------
    Focuses exclusively on generating clean Python code implementations based on
    the technical notes compiled by the Researcher Agent.
    """
    print("\n---> [Coder Agent] Writing Python implementation...")
    notes = state.get("research_notes", "")
    prompt = f"Write clean, working Python code for '{state['task']}' based on these notes:\n{notes}"
    res = llm.invoke(prompt).content.strip()
    return {"code_output": res}


# ============================================================================
# STEP 3: DEFINE SUPERVISOR / ORCHESTRATOR NODE
# ============================================================================

def supervisor_agent(state: MultiAgentState) -> dict:
    """
    Central Supervisor Node:
    -----------------------
    Evaluates current state to determine which worker agent should execute next:
    - If research notes are missing -> selects 'researcher'
    - If research is complete but code is missing -> selects 'coder'
    - If both sub-tasks are complete -> selects 'FINISH'
    """
    print("\n---> [Supervisor Agent] Evaluating state progress & directing workflow...")
    research = state.get("research_notes", "")
    code = state.get("code_output", "")
    
    if not research:
        next_step = "researcher"
    elif not code:
        next_step = "coder"
    else:
        next_step = "FINISH"
        
    print(f"     Supervisor Router Decision -> Send to '{next_step}'")
    return {"next_worker": next_step}


def route_supervisor(state: MultiAgentState) -> Literal["researcher", "coder", "FINISH"]:
    """Conditional Edge Router reading state['next_worker']."""
    return state.get("next_worker", "FINISH")


# ============================================================================
# STEP 4: BUILD MULTI-AGENT GRAPH ARCHITECTURE
# ============================================================================
builder = StateGraph(MultiAgentState)

# 1. Register Supervisor and Worker Nodes
builder.add_node("supervisor", supervisor_agent)
builder.add_node("researcher", researcher_agent)
builder.add_node("coder", coder_agent)

# 2. Entrypoint: START -> supervisor
builder.add_edge(START, "supervisor")

# 3. Dynamic Conditional Routing from Supervisor to Workers or FINISH
builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher": "researcher",
        "coder": "coder",
        "FINISH": END
    }
)

# 4. Workers route back to Supervisor after completing their task!
builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")

app = builder.compile()


# ============================================================================
# STEP 5: EXECUTION DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 07 - Lesson 02: Supervisor Multi-Agent System ")
    print("=" * 65)

    res = app.invoke({"task": "Async HTTP fetch utility using aiohttp"})

    print("\n" + "=" * 65)
    print("--- FINAL MULTI-AGENT OUTPUT ---")
    print("=" * 65)
    print(f"[Research Notes]:\n{res['research_notes']}\n")
    print("-" * 65)
    print(f"[Generated Code]:\n{res['code_output']}")
    print("=" * 65)
