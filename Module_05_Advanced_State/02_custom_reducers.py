"""
Module 05 - Lesson 02: Custom State Reducer Functions
=====================================================================
Demonstrates building custom state reducer functions in LangGraph using
`typing.Annotated` for advanced state update logic such as list deduplication
and sliding-window event log capping.

Key Concepts Demonstrated:
---------------------------
1. LangGraph State Reducers: Control how field values are merged when nodes
   return state updates (default behavior without reducer is overwrite).
2. Custom Deduplication Reducer (`deduplicate_reducer`): Merges new items into
   an existing list while filtering out duplicate values and preserving order.
3. Custom Sliding Window Reducer (`sliding_window_reducer`): Appends new items
   to a list while capping maximum length to N entries (e.g., latest 3 events).
4. `typing.Annotated` Field Definitions: Attaches reducer functions directly to
   state fields in `TypedDict` or `BaseModel` schemas.
"""

import os
import sys
import warnings
import logging
from typing import TypedDict, Annotated

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
# STEP 1: DEFINE CUSTOM REDUCER FUNCTIONS
# ============================================================================
# A custom reducer accepts two arguments:
#   - `existing`: The current state value accumulated so far.
#   - `new`: The update returned by the executing graph node.
# It returns the newly computed state value.

def deduplicate_reducer(existing: list[str], new: list[str]) -> list[str]:
    """
    Custom Reducer 1: Deduplication Reducer
    ---------------------------------------
    Combines existing and incoming list items while filtering out duplicate
    elements and preserving original insertion sequence order.
    """
    combined = list(existing)
    for item in new:
        if item not in combined:
            combined.append(item)
    return combined


def sliding_window_reducer(existing: list[str], new: list[str]) -> list[str]:
    """
    Custom Reducer 2: Sliding Window Reducer (Max Log Capping)
    ----------------------------------------------------------
    Appends new event items to existing entries but caps total list length
    to the last 3 items. Essential for managing token limits in conversation logs.
    """
    combined = existing + new
    # Slice to retain only the last 3 elements
    return combined[-3:]


# ============================================================================
# STEP 2: DEFINE GRAPH STATE USING CUSTOM REDUCERS
# ============================================================================
# We use typing.Annotated[Type, ReducerFunction] to bind reducers to fields.

class AdvancedReducerState(TypedDict):
    # Field 1: Uses deduplicate_reducer to maintain a list of unique tags
    tags: Annotated[list[str], deduplicate_reducer]
    
    # Field 2: Uses sliding_window_reducer to keep only the 3 most recent events
    recent_events: Annotated[list[str], sliding_window_reducer]


# ============================================================================
# STEP 3: DEFINE WORKFLOW NODES
# ============================================================================

def node_alpha(state: AdvancedReducerState) -> dict:
    """Node 1: Emits initial tags and event logs."""
    print("\n---> [Node Alpha] Emitting initial tags and events...")
    return {
        "tags": ["python", "ai", "graph"],
        "recent_events": ["Event 1", "Event 2"]
    }


def node_beta(state: AdvancedReducerState) -> dict:
    """
    Node 2: Emits overlapping tags ('ai' is duplicate) and additional events.
    The custom reducers will deduplicate 'ai' and accumulate events.
    """
    print("\n---> [Node Beta] Emitting overlapping tags and additional events...")
    return {
        "tags": ["ai", "langgraph", "llm"],  # 'ai' is already present in state
        "recent_events": ["Event 3", "Event 4"]  # Total events now exceeds 3
    }


def node_gamma(state: AdvancedReducerState) -> dict:
    """
    Node 3: Emits further tags ('python' is duplicate) and another event log.
    The custom reducers filter 'python' and cap event log length to max 3 entries.
    """
    print("\n---> [Node Gamma] Emitting final tags and event log...")
    return {
        "tags": ["python", "advanced"],  # 'python' is already present in state
        "recent_events": ["Event 5"]  # Triggers sliding window truncation
    }


# ============================================================================
# STEP 4: BUILD AND COMPILE GRAPH
# ============================================================================
builder = StateGraph(AdvancedReducerState)

# Register node functions
builder.add_node("alpha", node_alpha)
builder.add_node("beta", node_beta)
builder.add_node("gamma", node_gamma)

# Define sequential execution flow: START -> alpha -> beta -> gamma -> END
builder.add_edge(START, "alpha")
builder.add_edge("alpha", "beta")
builder.add_edge("beta", "gamma")
builder.add_edge("gamma", END)

# Compile the graph
app = builder.compile()


# ============================================================================
# STEP 5: EXECUTION DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 05 - Lesson 02: Custom Reducers (Deduplication & Sliding Window) ")
    print("=" * 65)

    # Execute graph starting with empty list defaults
    res = app.invoke({"tags": [], "recent_events": []})

    # Display final aggregated state values
    print("\n" + "=" * 65)
    print("--- Final State Inspection ---")
    print("=" * 65)
    
    print("\n1. Final Deduplicated Tags List:")
    print(f"   {res['tags']}")
    print("   [Note: 'ai' and 'python' were emitted multiple times but appear only once]")
    
    print("\n2. Final Sliding Window Events (Max 3 Entries Retained):")
    print(f"   {res['recent_events']}")
    print("   [Note: Earlier 'Event 1' and 'Event 2' were pruned as new events arrived]")
    print("=" * 65)
