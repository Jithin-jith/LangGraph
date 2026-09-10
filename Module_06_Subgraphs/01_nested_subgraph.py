"""
Module 06 - Lesson 01: Nested Subgraphs & Modular Architecture
=====================================================================
Demonstrates how to break complex graph workflows into modular, reusable
child `StateGraph` instances and embed them as nodes directly inside a parent graph.

Key Concepts Demonstrated:
---------------------------
1. Subgraph Modularization: Compiling isolated child graphs (`refinement_subgraph`)
   that can be reused across different parent workflows.
2. Direct Subgraph Embedding: Using `parent_builder.add_node("subgraph_name", compiled_subgraph)`
   to treat an entire subgraph as a single node inside the parent execution flow.
3. Shared State Inheritance: When parent and child graphs share identical state keys
   (e.g., `draft`, `critique`, `revised_draft`), state automatically flows between them.
4. Hierarchical Execution: The parent graph executes up to the subgraph node, delegates
   control to the child graph until completion, and then resumes parent graph execution.
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

# Initialize Gemini LLM for pitch generation and essay editing
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

# ============================================================================
# STEP 1: DEFINE CHILD SUBGRAPH (ESSAY REFINEMENT WORKFLOW)
# ============================================================================
# Subgraphs encapsulate modular sub-workflows. This child graph takes an initial
# draft, critiques it, and generates a polished revised draft.

class SubgraphState(TypedDict):
    draft: str          # Input draft text to evaluate
    critique: str       # Generated evaluation feedback
    revised_draft: str  # Polished output text

def generate_critique_node(state: SubgraphState) -> dict:
    """Child Node 1: Analyzes input draft and generates constructive critique."""
    print("   [Child Subgraph] Step A: Evaluating Draft with Gemini...")
    prompt = f"Critique this draft in 2 concise sentences highlighting strengths and flaws:\n{state['draft']}"
    res = llm.invoke(prompt).content
    return {"critique": res.strip()}

def revise_draft_node(state: SubgraphState) -> dict:
    """Child Node 2: Rewrites draft incorporating feedback from critique."""
    print("   [Child Subgraph] Step B: Polishing Draft based on Critique...")
    prompt = f"Improve this draft based on critique.\nDraft: {state['draft']}\nCritique: {state['critique']}"
    res = llm.invoke(prompt).content
    return {"revised_draft": res.strip()}

# Build and compile the Child Subgraph
sub_builder = StateGraph(SubgraphState)
sub_builder.add_node("generate_critique", generate_critique_node)
sub_builder.add_node("revise_draft", revise_draft_node)

# Define child graph execution flow
sub_builder.add_edge(START, "generate_critique")
sub_builder.add_edge("generate_critique", "revise_draft")
sub_builder.add_edge("revise_draft", END)

# Compile child subgraph into an executable unit
refinement_subgraph = sub_builder.compile()


# ============================================================================
# STEP 2: DEFINE PARENT GRAPH (PRODUCT PITCH GENERATOR)
# ============================================================================
# Parent graph schema contains all keys used in parent processing plus keys shared
# with the child subgraph (`draft`, `critique`, `revised_draft`).

class ParentState(TypedDict):
    product_name: str    # Initial product topic
    draft: str           # Generated initial pitch draft
    critique: str        # Shared with child subgraph
    revised_draft: str   # Output from child subgraph
    final_pitch: str     # Formatted final output

def initial_pitch_node(state: ParentState) -> dict:
    """Parent Node 1: Writes initial unrefined product pitch draft."""
    print("---> [Parent Graph] Step 1: Writing Initial Product Pitch...")
    prompt = f"Write a one-paragraph elevator pitch for a product named '{state['product_name']}'."
    res = llm.invoke(prompt).content
    return {"draft": res.strip()}

def finalize_pitch_node(state: ParentState) -> dict:
    """Parent Node 2: Formats the revised pitch for final delivery."""
    print("---> [Parent Graph] Step 3: Formatting Final Delivery Pitch...")
    return {"final_pitch": f"🔥 FINAL APPROVED PITCH 🔥\n{state['revised_draft']}"}

# Construct the Parent Graph
parent_builder = StateGraph(ParentState)

# Add regular parent node functions
parent_builder.add_node("initial_pitch", initial_pitch_node)

# EMBED COMPILED CHILD SUBGRAPH DIRECTLY AS A NODE IN PARENT GRAPH!
parent_builder.add_node("refinement_subgraph", refinement_subgraph)

parent_builder.add_node("finalize_pitch", finalize_pitch_node)

# Define parent routing: START -> initial_pitch -> refinement_subgraph -> finalize_pitch -> END
parent_builder.add_edge(START, "initial_pitch")
parent_builder.add_edge("initial_pitch", "refinement_subgraph")
parent_builder.add_edge("refinement_subgraph", "finalize_pitch")
parent_builder.add_edge("finalize_pitch", END)

# Compile parent graph
parent_graph = parent_builder.compile()


# ============================================================================
# STEP 3: EXECUTION DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 06 - Lesson 01: Nested Subgraph Composition ")
    print("=" * 65)

    # Invoke parent graph with initial product input
    res = parent_graph.invoke({"product_name": "NeuroFlow AI Keyboard"})
    
    print("\n" + "=" * 65)
    print("--- Final Output Pitch ---")
    print("=" * 65)
    print(res["final_pitch"])
    print("=" * 65)
