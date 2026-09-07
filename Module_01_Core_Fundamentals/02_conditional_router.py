"""
Module 01 - Lesson 02: Conditional Router & Dynamic Branching
Demonstrates dynamic routing using add_conditional_edges to direct state to domain-specific expert nodes.
"""

import os
import sys
import warnings
import logging
from typing import TypedDict, Literal

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
# STEP 1: DEFINE GRAPH STATE
# ============================================================================
# The state is a shared dictionary (schema) passed from node to node.
# - user_query: The incoming prompt from the user.
# - category: Set by classifier node ("code", "math", or "general").
# - response: The final answer produced by whichever expert node runs.
class RouterState(TypedDict):
    user_query: str
    category: str
    response: str

# Temperature=0.0 ensures deterministic LLM classification outputs
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)


# ============================================================================
# STEP 2: DEFINE ROUTER & WORKER NODES
# ============================================================================

def classify_intent_node(state: RouterState) -> dict:
    """
    Node 1: Intent Classifier Node
    ------------------------------
    Analyzes the user's input query using Gemini and classifies it into
    one of three categories: 'code', 'math', or 'general'.
    Returns a dictionary updating state['category'].
    """
    print("\n---> [Node: Classifier] Analyzing user query intent...")
    query = state["user_query"]
    prompt = (
        f"Classify the following query into exactly one category: [code, math, general].\n"
        f"Query: {query}\n"
        f"Respond ONLY with one word: code, math, or general."
    )
    # Get classification from LLM
    res = llm.invoke(prompt).content.strip().lower()
    
    # Fallback safety check if LLM outputs unexpected text
    if res not in ["code", "math", "general"]:
        res = "general"
        
    print(f"     Classified Category: '{res}'")
    return {"category": res}


def code_expert_node(state: RouterState) -> dict:
    """
    Worker Node A: Code Expert
    --------------------------
    Executes only if the query is routed to 'code'. Generates Python code.
    """
    print("---> [Node: Code Expert] Generating Python code solution...")
    res = llm.invoke(f"Write Python code to solve: {state['user_query']}")
    return {"response": f"[CODE SOLUTION]\n{res.content}"}


def math_expert_node(state: RouterState) -> dict:
    """
    Worker Node B: Math Expert
    --------------------------
    Executes only if the query is routed to 'math'. Provides step-by-step math.
    """
    print("---> [Node: Math Expert] Solving math problem step-by-step...")
    res = llm.invoke(f"Solve this math problem step-by-step: {state['user_query']}")
    return {"response": f"[MATH SOLUTION]\n{res.content}"}


def general_expert_node(state: RouterState) -> dict:
    """
    Worker Node C: General Expert
    -----------------------------
    Executes only if the query is routed to 'general'. Provides concise text answer.
    """
    print("---> [Node: General Expert] Answering general knowledge question...")
    res = llm.invoke(f"Answer concisely: {state['user_query']}")
    return {"response": f"[GENERAL ANSWER]\n{res.content}"}


# ============================================================================
# STEP 3: CONDITIONAL ROUTING FUNCTION
# ============================================================================
def route_by_category(state: RouterState) -> Literal["code_expert", "math_expert", "general_expert"]:
    """
    Conditional Routing Function (Decision Maker)
    ---------------------------------------------
    This function DOES NOT modify state. Instead, it inspects state['category']
    and returns a string key identifying which node the graph should move to next.

    Return values map directly to node names registered in StateGraph:
    - 'code'    -> returns 'code_expert'
    - 'math'    -> returns 'math_expert'
    - 'general' -> returns 'general_expert'
    """
    category = state.get("category", "general")
    if category == "code":
        return "code_expert"
    elif category == "math":
        return "math_expert"
    else:
        return "general_expert"


# ============================================================================
# STEP 4: BUILD AND COMPILE THE STATEGRAPH
# ============================================================================
builder = StateGraph(RouterState)

# 1. Register all nodes in the graph
builder.add_node("classifier", classify_intent_node)
builder.add_node("code_expert", code_expert_node)
builder.add_node("math_expert", math_expert_node)
builder.add_node("general_expert", general_expert_node)

# 2. Add starting edge (START -> classifier)
builder.add_edge(START, "classifier")

# 3. Add CONDITIONAL EDGES from classifier node
#    Parameters:
#    - source_node: "classifier" (Where the decision happens)
#    - path: route_by_category (Function that returns the destination key)
#    - path_map (optional dictionary): Maps routing function outputs to node names
builder.add_conditional_edges(
    "classifier",
    route_by_category,
    {
        "code_expert": "code_expert",
        "math_expert": "math_expert",
        "general_expert": "general_expert",
    }
)

# 4. Connect all expert worker nodes to graph termination (END)
builder.add_edge("code_expert", END)
builder.add_edge("math_expert", END)
builder.add_edge("general_expert", END)

# 5. Compile graph into an executable Runnable application
app = builder.compile()


# ============================================================================
# STEP 5: EXECUTION DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 01 - Lesson 02: Dynamic Routing with add_conditional_edges ")
    print("=" * 65)

    test_queries = [
        "How do I sort a list of dictionaries by key in Python?",
        "What is the integral of 3x^2 + 2x dx?",
        "What is the capital of France?"
    ]

    for idx, query in enumerate(test_queries, 1):
        print("\n" + "=" * 65)
        print(f"Test Query #{idx}: {query}")
        print("=" * 65)
        
        # Invoke the graph with the user query
        final_state = app.invoke({"user_query": query})
        
        print("\n[Final Output]:")
        print(final_state["response"])
        break
