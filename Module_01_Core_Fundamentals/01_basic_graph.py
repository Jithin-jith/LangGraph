import os
import warnings
import logging
from typing import TypedDict
from dotenv import load_dotenv

# Suppress Google GenAI SDK Automatic Function Calling (AFC) recommendation notice
warnings.filterwarnings("ignore")
logging.getLogger("google.genai").setLevel(logging.ERROR)

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

# Load environment variables (.env file)
load_dotenv()

# Step 1: Define the Graph State
class GraphState(TypedDict):
    topic: str
    greeting: str
    fact: str
    final_output: str

# Initialize Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# Step 2: Define Node Functions
def greeting_node(state: GraphState) -> dict:
    """Generates a friendly welcome greeting for the topic."""
    print("---> Executing Greeting Node")
    topic = state.get("topic", "AI")
    prompt = f"Write a one-sentence enthusiastic welcome greeting for someone learning about {topic}."
    response = llm.invoke(prompt)
    print(f"\n--- Greeting Node Response ---")
    print(response.content.strip())
    return {"greeting": response.content.strip()}

def fact_node(state: GraphState) -> dict:
    """Generates a fun fact about the topic."""
    print("---> Executing Fact Node")
    topic = state.get("topic", "AI")
    prompt = f"Provide one interesting fun fact about {topic} in two sentences."
    response = llm.invoke(prompt)
    print(f"\n--- Fact Node Response ---")
    print(response.content.strip())
    return {"fact": response.content.strip()}

def summarize_node(state: GraphState) -> dict:
    """Combines greeting and fact into a formatted final output."""
    print("---> Executing Summarize Node")
    greeting = state["greeting"]
    fact = state["fact"]
    combined = f"=== WELCOME ===\n{greeting}\n\n=== DID YOU KNOW? ===\n{fact}"
    return {"final_output": combined}

# Step 3: Build the State Graph
workflow = StateGraph(GraphState)

# Add Nodes to Graph
workflow.add_node("greeting_node", greeting_node)
workflow.add_node("fact_node", fact_node)
workflow.add_node("summarize_node", summarize_node)

# Add Edges (Linear Control Flow)
workflow.add_edge(START, "greeting_node")
workflow.add_edge("greeting_node", "fact_node")
workflow.add_edge("fact_node", "summarize_node")
workflow.add_edge("summarize_node", END)

# Step 4: Compile the Graph
app = workflow.compile()

# Step 5: Execute the Graph
if __name__ == "__main__":
    print("=" * 60)
    print(" Module 01 - Lesson 01: Basic Linear StateGraph with Gemini ")
    print("=" * 60)

    initial_input = {"topic": "Quantum Computing"}
    print(f"Input Topic: {initial_input['topic']}\n")

    result = app.invoke(initial_input)

    print("\n--- Final Graph State Result ---")
    print(result["final_output"])
