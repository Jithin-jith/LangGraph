"""
Module 01 - Lesson 03: State Reducers & Accumulation
Demonstrates using Annotated list state reducers (operator.add) to append values instead of overwriting state.
"""

import os
import sys
import warnings
import logging
from typing import TypedDict, Annotated
from operator import add

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Filter out lower-level SDK warnings written directly to sys.stderr
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
# STEP 1: DEFINE GRAPH STATE WITH REDUCERS
# ============================================================================
# In LangGraph:
# - Default field behavior: Returning a key in a node's returned dict OVERWRITES existing value.
# - Reducer field behavior: Typing a list with `Annotated[list[str], add]` APPENDS new list
#   elements to the existing list rather than overwriting it!
class ReducerState(TypedDict):
    task: str
    
    # 1. Standard Overwriting key: Each node return replaces 'last_thought'
    last_thought: str
    
    # 2. Reducer Appending key: `add` operator appends returned items to 'history_logs'
    history_logs: Annotated[list[str], add]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)


# ============================================================================
# STEP 2: DEFINE GRAPH NODES
# ============================================================================

def step_one_node(state: ReducerState) -> dict:
    """
    Node 1: Task Initialization Node
    --------------------------------
    Appends an initialization log to history_logs and sets initial last_thought.
    """
    print("---> [Node 1] Initiating Task...")
    task = state["task"]
    log_entry = f"Node 1 started processing task: '{task}'"
    return {
        "last_thought": "Node 1 finished initializing environment.",
        "history_logs": [log_entry] # Appended to history_logs list
    }


def step_two_node(state: ReducerState) -> dict:
    """
    Node 2: Content Generation Node
    -------------------------------
    Overwrites last_thought with tip result, and appends generation log to history_logs.
    """
    print("---> [Node 2] Generating Learning Tip with Gemini...")
    task = state["task"]
    prompt = f"Give a 1-sentence quick tip regarding: {task}"
    tip = llm.invoke(prompt).content.strip()
    log_entry = f"Node 2 generated tip: {tip}"
    return {
        "last_thought": f"Node 2 created tip: '{tip}'",
        "history_logs": [log_entry] # Appended to history_logs list via reducer
    }


def step_three_node(state: ReducerState) -> dict:
    """
    Node 3: Finalization Node
    -------------------------
    Overwrites last_thought with final status, and appends completion log.
    """
    print("---> [Node 3] Finalizing Execution...")
    log_entry = "Node 3 completed all validation and aggregation steps."
    return {
        "last_thought": "Node 3 finished all workflow tasks successfully.",
        "history_logs": [log_entry] # Appended to history_logs list
    }


# ============================================================================
# STEP 3: BUILD AND COMPILE THE STATEGRAPH
# ============================================================================
builder = StateGraph(ReducerState)

builder.add_node("step_one", step_one_node)
builder.add_node("step_two", step_two_node)
builder.add_node("step_three", step_three_node)

builder.add_edge(START, "step_one")
builder.add_edge("step_one", "step_two")
builder.add_edge("step_two", "step_three")
builder.add_edge("step_three", END)

app = builder.compile()


# ============================================================================
# STEP 4: EXECUTE AND DEMONSTRATE REDUCER MECHANICS
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 01 - Lesson 03: State Reducers & Value Accumulation ")
    print("=" * 65)

    # Initial state passed into app.invoke()
    initial_input = {
        "task": "Learning LangGraph State Management",
        "history_logs": ["System session launched"] # Initial item in reducer list
    }

    res = app.invoke(initial_input)

    print("\n" + "=" * 65)
    print("--- State Reduction Analysis ---")
    print("=" * 65)
    print(f"1. Overwritten Field ('last_thought'):\n   -> '{res['last_thought']}'")
    print(f"\n2. Accumulated Field with Reducer ('history_logs'):")
    for idx, entry in enumerate(res['history_logs'], 1):
        print(f"   {idx}. {entry}")
