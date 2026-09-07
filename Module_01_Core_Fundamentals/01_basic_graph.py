"""
Module 01 - Lesson 01: Basic Linear StateGraph
Demonstrates building a basic sequential StateGraph using TypedDict state schema and Google Gemini LLM nodes.
"""

import os
import sys
import warnings
import logging
from typing import TypedDict

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

# Load environment variables (.env file containing GOOGLE_API_KEY)
load_dotenv()

# ============================================================================
# STEP 1: DEFINE THE GRAPH STATE SCHEMA
# ============================================================================
# In LangGraph, state is a shared Python dictionary schema that flows through every node.
# Each node receives the current state snapshot, performs computation/LLM calls, 
# and returns a dictionary with state updates.
class GraphState(TypedDict):
    topic: str          # The topic provided as initial input
    greeting: str       # Generated welcome greeting string
    fact: str           # Generated fun fact string
    final_output: str   # Combined final summary formatted for display

# Initialize Google Gemini 2.5 Flash model with temperature=0.7 for creative responses
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


# ============================================================================
# STEP 2: DEFINE GRAPH NODE FUNCTIONS
# ============================================================================

def greeting_node(state: GraphState) -> dict:
    """
    Node 1: Welcome Greeting Generator
    ---------------------------------
    Receives current state, reads state['topic'], calls Gemini LLM to generate
    an enthusiastic welcome greeting, and returns {'greeting': response}.
    """
    print("---> Executing Greeting Node...")
    topic = state.get("topic", "AI")
    prompt = f"Write a one-sentence enthusiastic welcome greeting for someone learning about {topic}."
    response = llm.invoke(prompt)
    
    greeting_text = response.content.strip()
    print(f"\n--- Greeting Node Response ---\n{greeting_text}")
    return {"greeting": greeting_text}


def fact_node(state: GraphState) -> dict:
    """
    Node 2: Fun Fact Generator
    -------------------------
    Receives state (which now includes state['greeting']), calls Gemini LLM to
    generate an interesting fun fact about the topic, and returns {'fact': response}.
    """
    print("\n---> Executing Fact Node...")
    topic = state.get("topic", "AI")
    prompt = f"Provide one interesting fun fact about {topic} in two sentences."
    response = llm.invoke(prompt)
    
    fact_text = response.content.strip()
    print(f"\n--- Fact Node Response ---\n{fact_text}")
    return {"fact": fact_text}


def summarize_node(state: GraphState) -> dict:
    """
    Node 3: Final Aggregator Node
    ----------------------------
    Combines state['greeting'] and state['fact'] into a structured final text string.
    Returns {'final_output': combined_string}.
    """
    print("\n---> Executing Summarize Node...")
    greeting = state["greeting"]
    fact = state["fact"]
    combined = f"=== WELCOME ===\n{greeting}\n\n=== DID YOU KNOW? ===\n{fact}"
    return {"final_output": combined}


# ============================================================================
# STEP 3: BUILD AND COMPILE THE STATEGRAPH
# ============================================================================
# Instantiate StateGraph with our custom TypedDict schema
workflow = StateGraph(GraphState)

# 1. Add all nodes to graph builder (node_name -> node_function)
workflow.add_node("greeting_node", greeting_node)
workflow.add_node("fact_node", fact_node)
workflow.add_node("summarize_node", summarize_node)

# 2. Add sequential linear edges (START -> greeting -> fact -> summarize -> END)
workflow.add_edge(START, "greeting_node")
workflow.add_edge("greeting_node", "fact_node")
workflow.add_edge("fact_node", "summarize_node")
workflow.add_edge("summarize_node", END)

# 3. Compile graph into an executable Runnable application
app = workflow.compile()


# ============================================================================
# STEP 4: EXECUTE THE GRAPH
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" Module 01 - Lesson 01: Basic Linear StateGraph with Gemini ")
    print("=" * 60)

    # Initial input dictionary matching GraphState keys
    initial_input = {"topic": "Quantum Computing"}
    print(f"Input Topic: {initial_input['topic']}\n")

    # Run the graph synchronously from START to END
    result = app.invoke(initial_input)

    print("\n" + "=" * 60)
    print("--- Final Graph State Result ---")
    print("=" * 60)
    print(result["final_output"])
