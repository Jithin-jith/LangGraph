"""
Module 08 - Lesson 01: Fan-Out / Fan-In Parallel Graph Execution
=====================================================================
Demonstrates executing static parallel graph branches concurrently and
aggregating results using a fan-in reducer node.

Key Concepts Demonstrated:
---------------------------
1. Static Fan-Out Routing: Connecting `START` to multiple analyst nodes
   (`financial_analyst`, `tech_analyst`, `competitor_analyst`) executes all branches in parallel.
2. Accumulative Reducers: Using `Annotated[list[str], add]` to collect and append
   updates returned by parallel nodes without overwriting.
3. Fan-In Aggregation: Connecting all parallel analyst nodes into a single `aggregator_node`
   ensures execution waits until ALL parallel branches complete before compiling the master report.
4. Latency Optimization: Running non-dependent tasks concurrently drastically reduces total execution time.
"""

import os
import sys
import warnings
import logging
from typing import TypedDict, Annotated
from operator import add

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
# STEP 1: DEFINE GRAPH STATE WITH LIST REDUCER
# ============================================================================
class ParallelState(TypedDict):
    company_name: str
    # `Annotated[list[str], add]` accumulates outputs from parallel analyst nodes
    branch_results: Annotated[list[str], add]
    consolidated_report: str

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)


# ============================================================================
# STEP 2: DEFINE PARALLEL ANALYST NODES (FAN-OUT BRANCHES)
# ============================================================================

def financial_analyst(state: ParallelState) -> dict:
    """Parallel Branch A: Financial & Earnings Analysis."""
    print("   [Branch A] Analyzing Financial Outlook...")
    res = llm.invoke(f"Provide 1 bullet point on financial outlook for: {state['company_name']}").content.strip()
    return {"branch_results": [f"FINANCIAL: {res}"]}


def tech_analyst(state: ParallelState) -> dict:
    """Parallel Branch B: Tech & Product Stack Analysis."""
    print("   [Branch B] Analyzing Tech & Product Stack...")
    res = llm.invoke(f"Provide 1 bullet point on tech stack for: {state['company_name']}").content.strip()
    return {"branch_results": [f"TECH STACK: {res}"]}


def competitor_analyst(state: ParallelState) -> dict:
    """Parallel Branch C: Market Competitors Analysis."""
    print("   [Branch C] Analyzing Market Competitors...")
    res = llm.invoke(f"Provide 1 bullet point on main competitors for: {state['company_name']}").content.strip()
    return {"branch_results": [f"COMPETITORS: {res}"]}


# ============================================================================
# STEP 3: DEFINE AGGREGATOR NODE (FAN-IN JOIN NODE)
# ============================================================================

def aggregator_node(state: ParallelState) -> dict:
    """
    Fan-In Aggregator Node:
    -----------------------
    Executes ONLY after ALL 3 parallel analyst branches finish.
    Consolidates state['branch_results'] into a master dossier report.
    """
    print("\n---> [Fan-In Aggregator] Merging Parallel Analysis Results...")
    findings = "\n".join(state["branch_results"])
    summary = f"=== COMPREHENSIVE DOSSIER: {state['company_name']} ===\n{findings}"
    return {"consolidated_report": summary}


# ============================================================================
# STEP 4: BUILD GRAPH WITH PARALLEL FAN-OUT & FAN-IN EDGES
# ============================================================================
builder = StateGraph(ParallelState)

# Register analyst nodes and aggregator
builder.add_node("financial_analyst", financial_analyst)
builder.add_node("tech_analyst", tech_analyst)
builder.add_node("competitor_analyst", competitor_analyst)
builder.add_node("aggregator", aggregator_node)

# FAN-OUT: START connects to all 3 analysts simultaneously!
builder.add_edge(START, "financial_analyst")
builder.add_edge(START, "tech_analyst")
builder.add_edge(START, "competitor_analyst")

# FAN-IN: All 3 analysts connect into the single aggregator node
builder.add_edge("financial_analyst", "aggregator")
builder.add_edge("tech_analyst", "aggregator")
builder.add_edge("competitor_analyst", "aggregator")

builder.add_edge("aggregator", END)

app = builder.compile()


# ============================================================================
# STEP 5: EXECUTION DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 08 - Lesson 01: Parallel Fan-Out & Fan-In Graph Execution ")
    print("=" * 65)

    res = app.invoke({"company_name": "Google / Alphabet Inc."})
    
    print("\n" + "=" * 65)
    print(res["consolidated_report"])
    print("=" * 65)
