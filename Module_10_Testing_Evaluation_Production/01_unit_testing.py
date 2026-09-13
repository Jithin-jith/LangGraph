"""
Module 10 - Lesson 01: Unit & Integration Testing Graphs
=====================================================================
Demonstrates Pytest unit testing for individual graph node functions in isolation
as well as end-to-end integration testing for compiled LangGraph state machines.

Key Concepts Demonstrated:
---------------------------
1. Isolated Node Unit Testing: Node functions (`clean_text_node`, `count_chars_node`)
   accept plain dictionary states, making them easy to unit test with pytest without
   compiling or running LLM workflows.
2. End-to-End Integration Testing: Testing compiled `StateGraph` applications
   (`build_test_graph()`) to verify complete state transformation pathways.
3. Test Automation: Running `pytest.main(["-v", __file__])` directly from Python.
"""

import sys
import os
import warnings
import pytest
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
        if self.original_stderr:
            self.original_stderr.write(msg)

    def flush(self):
        if self.original_stderr and hasattr(self.original_stderr, "flush"):
            self.original_stderr.flush()

    def __getattr__(self, name):
        return getattr(self.original_stderr, name)

sys.stderr = StderrFilter(sys.stderr)

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from langgraph.graph import StateGraph, START, END

# ============================================================================
# STEP 1: DEFINE GRAPH STATE FOR TESTING
# ============================================================================
class SampleGraphState(TypedDict):
    input_text: str       # Raw input string
    processed_text: str   # Cleaned text output
    char_count: int       # Character length count


# ============================================================================
# STEP 2: DEFINE PURE NODE LOGIC FUNCTIONS (ISOLATED UNIT TESTING)
# ============================================================================

def clean_text_node(state: SampleGraphState) -> dict:
    """Node 1: Strips leading/trailing whitespace and converts text to lowercase."""
    cleaned = state["input_text"].strip().lower()
    return {"processed_text": cleaned}


def count_chars_node(state: SampleGraphState) -> dict:
    """Node 2: Calculates character length of processed text."""
    return {"char_count": len(state["processed_text"])}


# ============================================================================
# STEP 3: BUILD GRAPH COMPILATION FACTORY
# ============================================================================
def build_test_graph():
    """Factory function compiling the test state machine."""
    builder = StateGraph(SampleGraphState)
    builder.add_node("clean_text", clean_text_node)
    builder.add_node("count_chars", count_chars_node)
    
    builder.add_edge(START, "clean_text")
    builder.add_edge("clean_text", "count_chars")
    builder.add_edge("count_chars", END)
    
    return builder.compile()


# ============================================================================
# STEP 4: PYTEST UNIT & INTEGRATION TEST SUITE
# ============================================================================

def test_clean_text_node():
    """Unit Test: Verifies clean_text_node in isolation without running graph."""
    input_state = {"input_text": "  Hello LangGraph World!  "}
    result = clean_text_node(input_state)
    assert result["processed_text"] == "hello langgraph world!"


def test_count_chars_node():
    """Unit Test: Verifies count_chars_node in isolation."""
    input_state = {"processed_text": "abc"}
    result = count_chars_node(input_state)
    assert result["char_count"] == 3


def test_full_graph_integration():
    """Integration Test: Verifies end-to-end state transitions across compiled graph."""
    app = build_test_graph()
    output = app.invoke({"input_text": "  LangGraph  "})
    assert output["processed_text"] == "langgraph"
    assert output["char_count"] == 9


# ============================================================================
# STEP 5: EXECUTION DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 10 - Lesson 01: Unit & Integration Testing Graphs ")
    print("=" * 65)
    print("Running Pytest Test Suite...")
    pytest.main(["-v", __file__])
