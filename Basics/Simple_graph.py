from langgraph.graph import StateGraph, END,START
from typing import TypedDict, Optional
import random
from IPython.display import display,Image

# Define State Schema
class DrinkState(TypedDict):
    time_of_day: Optional[str]
    drink_decision: Optional[str]

# Define Node Functions
def start_node(state: DrinkState) -> DrinkState:
    print("[Start Node] Starting decision process")
    return state


def check_time_of_day_node(state: DrinkState) -> DrinkState:
    # For demo, hardcode time_of_day
    if random.random() >= 0.5:
        time_of_day = "morning"  
    else :
         time_of_day = "evening"
    state["time_of_day"] = time_of_day
    print(f"[Check Time] It is {time_of_day}")
    return state

def decide_drink_node(state: DrinkState) -> DrinkState:
    if state["time_of_day"] == "morning":
        state["drink_decision"] = "Coffee"
    elif state["time_of_day"] == "evening":
        state["drink_decision"] = "Tea"
    else:
        state["drink_decision"] = "No Drink"
    print(f"[Decide Drink] Decision: {state['drink_decision']}")
    return state

def end_node(state: DrinkState) -> DrinkState:
    print(f"[End Node] Final decision: {state['drink_decision']}")
    return state

# Create the Graph
graph_builder = StateGraph(DrinkState)

# Add nodes
graph_builder.add_node("start", start_node)
graph_builder.add_node("check_time", check_time_of_day_node)
graph_builder.add_node("decide_drink", decide_drink_node)
graph_builder.add_node("end", end_node)

# Define Edges
graph_builder.add_edge(START,"start")
graph_builder.add_edge("start", "check_time")
graph_builder.add_edge("check_time", "decide_drink")
graph_builder.add_edge("decide_drink", "end")
graph_builder.add_edge("end", END)

# Compile graph
graph = graph_builder.compile()

# View graph
# display(Image(graph.get_graph().draw_mermaid_png()))

# Run
input_state = {}
final_state = graph.invoke(input_state)

