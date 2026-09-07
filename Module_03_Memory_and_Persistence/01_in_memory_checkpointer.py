import os
import sys
import warnings
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Suppress Google GenAI SDK Automatic Function Calling (AFC) recommendation notice
warnings.filterwarnings("ignore", message=".*automatic function calling.*")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ============================================================================
# STEP 1: INITIALIZE MODEL & AGENT NODE
# ============================================================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

def chatbot_node(state: MessagesState) -> dict:
    """Conversational node reading and appending to message history."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build state graph
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)


# ============================================================================
# STEP 2: ATTACH IN-MEMORY CHECKPOINTER
# ============================================================================
# MemorySaver stores state snapshots in memory keyed by `thread_id`.
# When checkpointer is attached, graph.invoke() automatically retrieves
# past state for the given thread_id!
checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)


# ============================================================================
# STEP 3: DEMONSTRATE THREAD ISOLATION
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 03 - Lesson 01: In-Memory Checkpointer & Thread Isolation ")
    print("=" * 65)

    # Thread 1: User Alice Configuration
    config_alice = {"configurable": {"thread_id": "thread-alice"}}
    
    print("\n--- Thread 1: Alice's Conversation ---")
    r1 = app.invoke({"messages": [HumanMessage(content="Hi, my favorite color is Cyan.")]}, config=config_alice)
    print(f"Alice Turn 1 response: {r1['messages'][-1].content}\n")

    # Turn 2: Alice asks question relying on Turn 1 memory
    r2 = app.invoke({"messages": [HumanMessage(content="What is my favorite color?")]}, config=config_alice)
    print(f"Alice Turn 2 response: {r2['messages'][-1].content}\n")

    # Thread 2: User Bob Configuration (Completely isolated state!)
    config_bob = {"configurable": {"thread_id": "thread-bob"}}
    
    print("--- Thread 2: Bob's Conversation (Isolated State) ---")
    r_bob = app.invoke({"messages": [HumanMessage(content="What is my favorite color?")]}, config=config_bob)
    print(f"Bob Turn 1 response: {r_bob['messages'][-1].content}")
