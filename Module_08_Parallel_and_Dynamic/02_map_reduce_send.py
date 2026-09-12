"""
Module 08 - Lesson 02: Dynamic Map-Reduce with LangGraph `Send` API
=====================================================================
Demonstrates dynamic parallel task dispatching using LangGraph's `Send` API.
A map router node dynamically instantiates worker tasks at runtime based on input
collection length, and a reduce node aggregates all worker outputs into a digest.

Key Concepts Demonstrated:
---------------------------
1. LangGraph `Send` API: `Send("target_node", {"arg": value})` dynamically spawns
   an individual worker task instance for each element in a collection.
2. Dynamic Fan-Out: Unlike static edges, `Send` API handles variable-length input
   lists (e.g. 3, 10, or 100 items) dynamically at runtime without modifying graph structure.
3. Independent Worker State: Each worker task receives its own isolated `WorkerTaskState` dict.
4. Reduce Aggregation Node: The `reduce_results_node` collects state updates returned
   by all worker tasks via the `summaries` list reducer.
"""

import sys
import os
import warnings
import logging
from typing import TypedDict, Annotated, List
from operator import add
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
from langgraph.types import Send

load_dotenv()

# ============================================================================
# STEP 1: DEFINE MASTER GRAPH STATE & WORKER STATE
# ============================================================================
# Master Graph State holds list of input subjects and aggregated summaries
class MapReduceState(TypedDict):
    subjects: List[str]                    # Input list of topics to map over
    summaries: Annotated[list[str], add]   # Accumulated outputs from worker tasks
    final_digest: str                      # Consolidated output digest

# Isolated state schema passed to each dynamically spawned worker task instance
class WorkerTaskState(TypedDict):
    subject: str

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)


# ============================================================================
# STEP 2: DYNAMIC MAP DISPATCH VIA SEND API
# ============================================================================
def spawn_map_tasks(state: MapReduceState) -> List[Send]:
    """
    Map Router Function:
    --------------------
    Reads state['subjects'] and returns a list of Send("target_node", worker_state) objects.
    LangGraph dynamically spawns N parallel worker task instances at runtime!
    """
    print(f"\n---> [Map Router] Spawning {len(state['subjects'])} dynamic worker tasks via Send API...")
    return [Send("summarize_subject_worker", {"subject": topic}) for topic in state["subjects"]]


def summarize_subject_worker(state: WorkerTaskState) -> dict:
    """
    Worker Task Node:
    -----------------
    Processes a single mapped subject item in parallel.
    Returns summary appended to master state['summaries'] via list reducer.
    """
    subj = state["subject"]
    print(f"   [Worker Task] Summarizing topic: '{subj}'...")
    res = llm.invoke(f"Write a 1-sentence summary of key concepts in: {subj}").content.strip()
    return {"summaries": [f"• {subj.upper()}: {res}"]}


def reduce_results_node(state: MapReduceState) -> dict:
    """
    Reduce Node:
    ------------
    Aggregates all worker outputs collected in state['summaries'] into a final master report.
    Executes only after all dynamically spawned worker tasks complete.
    """
    print("\n---> [Reduce Node] Aggregating all worker summaries into final digest...")
    all_summaries = "\n".join(state["summaries"])
    digest = f"=== EXECUTIVE TECH DIGEST ===\n{all_summaries}"
    return {"final_digest": digest}


# ============================================================================
# STEP 3: BUILD MAP-REDUCE GRAPH WITH SEND API
# ============================================================================
builder = StateGraph(MapReduceState)

# Register worker node and reduce node
builder.add_node("summarize_subject_worker", summarize_subject_worker)
builder.add_node("reduce_results", reduce_results_node)

# Dynamic Map Routing: START uses conditional edges with `spawn_map_tasks` returning Send objects
builder.add_conditional_edges(START, spawn_map_tasks, ["summarize_subject_worker"])

# Connect all dynamic workers to the reduce node
builder.add_edge("summarize_subject_worker", "reduce_results")
builder.add_edge("reduce_results", END)

app = builder.compile()


# ============================================================================
# STEP 4: EXECUTION DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 08 - Lesson 02: Dynamic Map-Reduce with LangGraph Send API ")
    print("=" * 65)

    input_topics = [
        "Vector Databases (Qdrant, Pinecone)",
        "Retrieval-Augmented Generation (RAG)",
        "LangGraph State Machines",
        "Autonomous AI Agents"
    ]

    res = app.invoke({"subjects": input_topics, "summaries": []})
    
    print("\n" + "=" * 65)
    print(res["final_digest"])
    print("=" * 65)
