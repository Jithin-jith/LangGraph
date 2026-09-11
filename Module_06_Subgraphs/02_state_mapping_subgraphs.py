"""
Module 06 - Lesson 02: Parent-Child State Mapping Wrappers
=====================================================================
Demonstrates how to integrate subgraphs when parent and child state schemas
have different key names and structures by using a state mapping wrapper node.

Key Concepts Demonstrated:
---------------------------
1. Schema Isolation: Preventing parent graph schemas (`EnterprisePipelineStates`)
   from polluting domain-specific subgraph schemas (`CodeReviewState`).
2. Input State Mapping: Converting `ParentState -> SubgraphState` before invoking
   the child subgraph.
3. Synchronous Subgraph Invocation: Calling `subgraph.invoke(subgraph_input)` inside
   the wrapper node function.
4. Output State Mapping: Extracting results from `SubgraphState` and formatting them
   back into `ParentState` schema keys.
"""

import os
import sys
import warnings
import logging
from typing import TypedDict

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

# Initialize Gemini LLM for code security and performance auditing
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# ============================================================================
# STEP 1: DEFINE SUBGRAPH WITH INDEPENDENT DOMAIN STATE SCHEMA
# ============================================================================
# Subgraph schema uses keys specific to code review ('code_snippet', 'security_vulnerabilities', 'performance_score')

class CodeReviewState(TypedDict):
    code_snippet: str               # Target code to audit
    security_vulnerabilities: str   # Identified security flaws
    performance_score: str          # Execution & complexity assessment

def check_security(state: CodeReviewState) -> dict:
    """Child Node 1: Audits code for SQL injection, XSS, and security vulnerabilities."""
    print("   [CodeReview Subgraph] Step A: Auditing Code Security...")
    prompt = f"Identify any potential security flaws in 1 sentence:\n{state['code_snippet']}"
    res = llm.invoke(prompt).content.strip()
    return {"security_vulnerabilities": res}

def check_performance(state: CodeReviewState) -> dict:
    """Child Node 2: Evaluates computational complexity and memory usage."""
    print("   [CodeReview Subgraph] Step B: Evaluating Performance & Complexity...")
    prompt = f"Evaluate execution speed and memory efficiency in 1 sentence:\n{state['code_snippet']}"
    res = llm.invoke(prompt).content.strip()
    return {"performance_score": res}

# Build and compile Code Review Subgraph
sub_builder = StateGraph(CodeReviewState)
sub_builder.add_node("check_security", check_security)
sub_builder.add_node("check_performance", check_performance)

# Connect child graph execution flow
sub_builder.add_edge(START, "check_security")
sub_builder.add_edge("check_security", "check_performance")
sub_builder.add_edge("check_performance", END)

# Compile subgraph
code_review_subgraph = sub_builder.compile()


# ============================================================================
# STEP 2: DEFINE PARENT GRAPH SCHEMA & WRAPPER TRANSLATION NODE
# ============================================================================
# Parent graph schema uses enterprise pipeline keys ('file_name', 'raw_python_code', 'audit_report')

class EnterprisePipelineStates(TypedDict):
    file_name: str         # File basename
    raw_python_code: str   # Source code contents
    audit_report: str      # Consolidated audit summary

def run_code_review_wrapper(state: EnterprisePipelineStates) -> dict:
    """
    State Mapping Wrapper Node
    ---------------------------
    Translates ParentState -> SubgraphState before invoking child graph,
    and maps SubgraphState -> ParentState after child graph completes.
    
    Why wrapper nodes are useful:
    - Allows subgraphs to stay decoupled and reusable across different parent apps.
    - Prevents schema pollution when parent state and child state use different key names.
    """
    print("---> [Parent Graph] Invoking Code Review Subgraph with state translation...")
    
    # 1. Input Mapping: Extract parent fields into subgraph input dict
    subgraph_input = {"code_snippet": state["raw_python_code"]}
    
    # 2. Synchronous Invocation of Subgraph
    subgraph_output = code_review_subgraph.invoke(subgraph_input)
    
    # 3. Output Mapping: Transform subgraph output keys back into parent state schema
    report = (
        f"FILE AUDITED: {state['file_name']}\n"
        f"Security Flaws: {subgraph_output['security_vulnerabilities']}\n"
        f"Performance   : {subgraph_output['performance_score']}"
    )
    return {"audit_report": report}

# Build Parent Graph
parent_builder = StateGraph(EnterprisePipelineStates)
parent_builder.add_node("code_reviewer", run_code_review_wrapper)
parent_builder.add_edge(START, "code_reviewer")
parent_builder.add_edge("code_reviewer", END)

app = parent_builder.compile()


# ============================================================================
# STEP 3: EXECUTION DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 06 - Lesson 02: State Mapping between Parent & Subgraphs ")
    print("=" * 65)

    sample_code = """
def fetch_user_data(user_id):
    query = "SELECT * FROM users WHERE id = '" + str(user_id) + "'"
    return db.execute(query)
"""

    # Run enterprise pipeline
    res = app.invoke({
        "file_name": "db_utils.py",
        "raw_python_code": sample_code
    })

    # Print final mapped report
    print("\n" + "=" * 65)
    print("--- Final Enterprise Audit Report ---")
    print("=" * 65)
    print(res["audit_report"])
    print("=" * 65)
