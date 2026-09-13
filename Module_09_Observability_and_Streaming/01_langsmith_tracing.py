"""
Module 09 - Lesson 01: LangSmith Telemetry & Graph Tracing
=====================================================================
Demonstrates integrating LangSmith observability and telemetry with LangGraph.
Setting environment variables enables automatic tracing of graph node execution times,
LLM call inputs/outputs, latency metrics, and token consumption statistics.

Key Concepts Demonstrated:
---------------------------
1. LangSmith Environment Setup: Configuring `LANGCHAIN_TRACING_V2="true"` and
   `LANGCHAIN_PROJECT` enables non-intrusive background telemetry streaming.
2. Step-by-Step Node Tracing: Every node execution (`analysis`, `synthesis`) is tracked
   in the trace visualizer with exact start/end timestamps.
3. Token & Cost Telemetry: Automatically captures prompt tokens, completion tokens,
   and model latency across Gemini API calls.
4. Non-Intrusive Integration: Zero code changes required in `StateGraph` definition;
   tracing operates transparently at the framework level.
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
# STEP 1: ENABLE LANGSMITH TELEMETRY & TRACING
# ============================================================================
# Setting LANGCHAIN_TRACING_V2="true" automatically streams execution telemetry,
# latency metrics, token consumption, and graph traces directly to LangSmith!
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "LangGraph-Gemini-Masterclass"
    print(" [Observability] LangSmith Tracing Active! Project: 'LangGraph-Gemini-Masterclass'")

class TraceState(TypedDict):
    input_prompt: str           # User question
    intermediate_analysis: str  # Output from analysis node
    final_output: str           # Consolidated recommendation

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


# ============================================================================
# STEP 2: DEFINE TRACED WORKFLOW NODES
# ============================================================================

def node_analysis(state: TraceState) -> dict:
    """Step 1: Performs core architectural analysis (Traced in LangSmith)."""
    print("\n---> [Node 1: analysis] Deconstructing problem (Traced in LangSmith)...")
    res = llm.invoke(f"Deconstruct this problem into 2 core components: {state['input_prompt']}").content
    return {"intermediate_analysis": res}


def node_synthesis(state: TraceState) -> dict:
    """Step 2: Synthesizes final recommendations (Traced in LangSmith)."""
    print("\n---> [Node 2: synthesis] Formulating final recommendations (Traced in LangSmith)...")
    res = llm.invoke(f"Based on analysis:\n{state['intermediate_analysis']}\nProvide final recommendation.").content
    return {"final_output": res}


# ============================================================================
# STEP 3: BUILD AND COMPILE GRAPH
# ============================================================================
builder = StateGraph(TraceState)
builder.add_node("analysis", node_analysis)
builder.add_node("synthesis", node_synthesis)

builder.add_edge(START, "analysis")
builder.add_edge("analysis", "synthesis")
builder.add_edge("synthesis", END)

app = builder.compile()


# ============================================================================
# STEP 4: EXECUTION DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 09 - Lesson 01: LangSmith Telemetry & Graph Tracing ")
    print("=" * 65)

    res = app.invoke({"input_prompt": "How to migrate a monolithic Python application to serverless micro-services?"})

    print("\n" + "=" * 65)
    print("--- Final Recommendation ---")
    print("=" * 65)
    print(res["final_output"])
    print("\n[Observability Note]: Open https://smith.langchain.com to view node timing traces & token metrics.")
    print("=" * 65)
