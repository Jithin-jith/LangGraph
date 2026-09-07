"""
Module 03 - Lesson 01: In-Memory Checkpointer & Thread Isolation
Demonstrates using MemorySaver for server-side state persistence and session isolation across thread IDs.
"""

import os
import sys
import warnings
import logging

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
    alice_msg_1 = "Hi, my favorite color is Cyan."
    print(f"Alice (User): {alice_msg_1}")
    r1 = app.invoke({"messages": [HumanMessage(content=alice_msg_1)]}, config=config_alice)
    print(f"AI Response: {r1['messages'][-1].content}\n")

    # Turn 2: Alice asks question relying on Turn 1 memory
    alice_msg_2 = "What is my favorite color?"
    print(f"Alice (User): {alice_msg_2}")
    r2 = app.invoke({"messages": [HumanMessage(content=alice_msg_2)]}, config=config_alice)
    print(f"AI Response: {r2['messages'][-1].content}\n")

    # Thread 2: User Bob Configuration (Completely isolated state!)
    config_bob = {"configurable": {"thread_id": "thread-bob"}}
    
    print("--- Thread 2: Bob's Conversation (Isolated State) ---")
    bob_msg_1 = "What is my favorite color?"
    print(f"Bob (User): {bob_msg_1}")
    r_bob = app.invoke({"messages": [HumanMessage(content=bob_msg_1)]}, config=config_bob)
    print(f"AI Response: {r_bob['messages'][-1].content}")
