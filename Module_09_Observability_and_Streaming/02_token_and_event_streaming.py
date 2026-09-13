"""
Module 09 - Lesson 02: Real-time Graph Output Streaming
=====================================================================
Demonstrates streaming graph execution events and state snapshots in real time.
Compares `stream_mode='updates'` (node step return dicts) vs `stream_mode='values'`
(full graph state snapshots after each node completion).

Key Concepts Demonstrated:
---------------------------
1. `stream_mode="updates"`: Yields dictionaries containing state changes returned
   by individual nodes as soon as each node completes execution.
2. `stream_mode="values"`: Yields complete cumulative graph state snapshots after
   each step in the execution pipeline.
3. User Experience Optimization: Streaming reduces perceived latency by giving users
   immediate visual progress as nodes process.
"""

import sys
import os
import warnings
import logging
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
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# ============================================================================
# STEP 1: DEFINE WORKFLOW NODES
# ============================================================================

def node_step_one(state: MessagesState) -> dict:
    """Step 1: Queries Gemini for AI researchers list."""
    print("\n---> [Node 1: researcher_node] Fetching AI Researchers...")
    res = llm.invoke([HumanMessage(content="Name 3 famous AI researchers and their top contributions.")])
    return {"messages": [res]}


def node_step_two(state: MessagesState) -> dict:
    """Step 2: Summarizes researchers list into a single sentence."""
    print("\n---> [Node 2: summarizer_node] Summarizing Output...")
    last_msg = state["messages"][-1].content
    res = llm.invoke([HumanMessage(content=f"Summarize this list in one sentence:\n{last_msg}")])
    return {"messages": [res]}


# ============================================================================
# STEP 2: BUILD GRAPH
# ============================================================================
builder = StateGraph(MessagesState)
builder.add_node("researcher_node", node_step_one)
builder.add_node("summarizer_node", node_step_two)

builder.add_edge(START, "researcher_node")
builder.add_edge("researcher_node", "summarizer_node")
builder.add_edge("summarizer_node", END)

app = builder.compile()


# ============================================================================
# STEP 3: DEMONSTRATE STREAMING MODES ('updates' vs 'values')
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 09 - Lesson 02: Real-time Graph Output Streaming ")
    print("=" * 65)

    # ------------------------------------------------------------------------
    # 1. Stream Mode: 'updates' -> Emits node state updates as each node finishes
    # ------------------------------------------------------------------------
    print("\n--- 1. STREAM MODE: 'updates' (Emits node return updates) ---")
    for event in app.stream({"messages": []}, stream_mode="updates"):
        for node_name, state_update in event.items():
            print(f"\n[STREAM EVENT - NODE FINISHED: '{node_name}']")
            snippet = state_update['messages'][-1].content[:120].replace('\n', ' ')
            print(f" Output Message Snippet: {snippet}...")

    # ------------------------------------------------------------------------
    # 2. Stream Mode: 'values' -> Emits complete cumulative state dictionary
    # ------------------------------------------------------------------------
    print("\n\n--- 2. STREAM MODE: 'values' (Emits full state after each step) ---")
    for step, state in enumerate(app.stream({"messages": []}, stream_mode="values")):
        num_msgs = len(state.get("messages", []))
        print(f"Step {step}: Total Messages in Cumulative State = {num_msgs}")
    
    print("\n" + "=" * 65)
