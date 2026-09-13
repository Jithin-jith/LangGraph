"""
Module 09 - Lesson 03: Fault Tolerance, Fallbacks & Error Resilience
=====================================================================
Demonstrates building resilient LangGraph workflows with automatic fallback routing.
If the primary LLM call encounters errors, rate limits, or network timeouts, the graph
dynamically routes execution to a secondary fallback model node.

Key Concepts Demonstrated:
---------------------------
1. Error Interception: Wrapping node logic in `try-except` blocks to prevent unhandled
   exceptions from crashing graph execution.
2. Fallback State Flags: Setting state variables (`fallback_used=True`) upon exception
   to trigger conditional fallback routing.
3. Model Redundancy: Seamlessly switching from primary model (`gemini-2.5-flash`) to
   backup fallback model (`gemini-1.5-flash`).
4. Production Graceful Degradation: Ensuring high system availability even during downstream
   API outages or rate limit spikes.
"""

import sys
import os
import warnings
import logging
from typing import TypedDict
from dotenv import load_dotenv

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
        if self.original_stderr:
            self.original_stderr.write(msg)

    def flush(self):
        if self.original_stderr and hasattr(self.original_stderr, "flush"):
            self.original_stderr.flush()

sys.stderr = StderrFilter(sys.stderr)

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ============================================================================
# STEP 1: DEFINE GRAPH STATE
# ============================================================================
class ResilientState(TypedDict):
    prompt: str             # User prompt text
    primary_response: str   # Output from primary model
    fallback_used: bool     # Flag indicating if fallback recovery was triggered
    final_output: str       # Consolidated final output

# Primary Model Instance (First Choice)
primary_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

# Backup Fallback Model Instance (Second Choice)
fallback_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)


# ============================================================================
# STEP 2: DEFINE WORKFLOW NODES WITH TRY-EXCEPT ERROR CATCHING
# ============================================================================

def primary_node(state: ResilientState) -> dict:
    """
    Primary Model Node:
    -------------------
    Attempts execution with primary model. Catches exceptions (e.g. rate limits,
    network dropouts, model overload) and sets state['fallback_used'] = True.
    """
    print("\n---> [Node 1: primary_node] Attempting Primary Model Execution...")
    try:
        if "TRIGGER_ERROR" in state["prompt"]:
            raise RuntimeError("Simulated Primary API rate limit / model overload exception!")
            
        res = primary_llm.invoke(state["prompt"]).content
        return {"primary_response": res, "fallback_used": False}
    except Exception as err:
        print(f"   ⚠️ [Primary Node Failed]: {err}")
        return {"primary_response": "", "fallback_used": True}


def fallback_recovery_node(state: ResilientState) -> dict:
    """
    Fallback Recovery Node:
    -----------------------
    Executes ONLY if primary model fails. Invokes backup fallback model.
    """
    print("---> [Node 2: fallback_recovery] Executing Backup Fallback Model...")
    res = fallback_llm.invoke(state["prompt"]).content
    return {"final_output": f"[RECOVERED VIA FALLBACK MODEL]:\n{res}"}


def standard_finish_node(state: ResilientState) -> dict:
    """Standard Finish Node: Executes when primary model succeeds."""
    print("---> [Node 2: standard_finish] Finalizing Primary Output...")
    return {"final_output": f"[PRIMARY MODEL SUCCESS]:\n{state['primary_response']}"}


# ============================================================================
# STEP 3: CONDITIONAL ROUTER (ROUTES TO FALLBACK OR STANDARD)
# ============================================================================
def route_resilience(state: ResilientState) -> str:
    """Routes based on state['fallback_used'] flag."""
    if state.get("fallback_used", False):
        print("   [Router Decision]: Primary failed -> Routing to fallback_recovery node.")
        return "fallback_recovery"
    print("   [Router Decision]: Primary succeeded -> Routing to standard_finish node.")
    return "standard_finish"


# ============================================================================
# STEP 4: BUILD GRAPH
# ============================================================================
builder = StateGraph(ResilientState)

# Register nodes
builder.add_node("primary_node", primary_node)
builder.add_node("fallback_recovery", fallback_recovery_node)
builder.add_node("standard_finish", standard_finish_node)

builder.add_edge(START, "primary_node")

# Conditional routing based on execution success/failure
builder.add_conditional_edges(
    "primary_node",
    route_resilience,
    {
        "fallback_recovery": "fallback_recovery",
        "standard_finish": "standard_finish"
    }
)
builder.add_edge("fallback_recovery", END)
builder.add_edge("standard_finish", END)

app = builder.compile()


# ============================================================================
# STEP 5: EXECUTION DEMO (SUCCESS vs ERROR FALLBACK)
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 09 - Lesson 03: Fault Tolerance, Fallbacks & Error Resilience ")
    print("=" * 65)

    print("\n--- Test 1: Normal Primary Execution ---")
    r1 = app.invoke({"prompt": "Explain the concept of idempotency in API design."})
    print(r1["final_output"][:160] + "...\n")

    print("--- Test 2: Error Triggered (Fallback Model Activated) ---")
    r2 = app.invoke({"prompt": "TRIGGER_ERROR: Explain the concept of idempotency in API design."})
    print(r2["final_output"][:160] + "...")
    print("=" * 65)
