"""
Module 03 - Lesson 04: Multi-User SQLite Persistent Chatbot
Demonstrates a multi-user chatbot application with distinct thread persistence and data isolation in SQLite.
"""

import os
import sys
import warnings
import sqlite3
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
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# ============================================================================
# STEP 1: INITIALIZE MODEL & GRAPH BUILDER
# ============================================================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

def chatbot_node(state: MessagesState) -> dict:
    """Conversational node reading and appending to state messages."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build basic state graph
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

db_path = "multi_user_checkpoints.sqlite"


# ============================================================================
# STEP 2: MULTI-USER SQLITE CHECKPOINTING & THREAD ISOLATION DEMO
# ============================================================================
def run_multi_user_demo():
    print("=" * 70)
    print(" Module 03 - Lesson 04: Multi-User SQLite Persistent Chatbot ")
    print("=" * 70)

    # Establish SQLite connection context manager
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        # SqliteSaver persists state checkpoints to disk in SQLite tables
        memory = SqliteSaver(conn)
        app = builder.compile(checkpointer=memory)

        # --------------------------------------------------------------------
        # HARDCODED USER CONFIGURATIONS (THREAD ISOLATION KEYS)
        # --------------------------------------------------------------------
        # User 1: Alice
        config_alice = {"configurable": {"thread_id": "user-alice-thread"}}
        
        # User 2: Bob
        config_bob = {"configurable": {"thread_id": "user-bob-thread"}}

        # ====================================================================
        # PHASE 1: INITIAL DATA INGESTION (SAVED TO SQLITE DB)
        # ====================================================================
        print("\n--- PHASE 1: Storing Distinct Data for Alice and Bob in SQLite ---")
        
        # Alice stores her data
        print("\n[User: Alice] -> Sending profile data...")
        res_a1 = app.invoke(
            {"messages": [HumanMessage(content="Hi! My name is Alice. My secret project is Project Alpha and my role is Team Lead.")]},
            config=config_alice
        )
        print(f"AI Response to Alice: {res_a1['messages'][-1].content}\n")

        # Bob stores his data
        print("[User: Bob] -> Sending profile data...")
        res_b1 = app.invoke(
            {"messages": [HumanMessage(content="Hello! My name is Bob. My secret project is Project Omega and my role is Security Specialist.")]},
            config=config_bob
        )
        print(f"AI Response to Bob: {res_b1['messages'][-1].content}\n")

        # ====================================================================
        # PHASE 2: DATA EXTRACTION FROM SQLITE DB PER USER THREAD
        # ====================================================================
        print("-" * 70)
        print("--- PHASE 2: Querying User Data Extracted From SQLite Database ---")
        print("-" * 70)

        # Query Alice's data from thread-alice
        print("\n[User: Alice] -> Querying her secret project & role...")
        res_a2 = app.invoke(
            {"messages": [HumanMessage(content="What is my secret project and role?")]},
            config=config_alice
        )
        print(f"AI Response (Extracted from Alice's SQLite State):\n{res_a2['messages'][-1].content}\n")

        # Query Bob's data from thread-bob
        print("[User: Bob] -> Querying his secret project & role...")
        res_b2 = app.invoke(
            {"messages": [HumanMessage(content="What is my secret project and role?")]},
            config=config_bob
        )
        print(f"AI Response (Extracted from Bob's SQLite State):\n{res_b2['messages'][-1].content}\n")

        # ====================================================================
        # PHASE 3: VERIFY THREAD ISOLATION
        # ====================================================================
        print("-" * 70)
        print("--- PHASE 3: Verifying Thread Privacy / Data Isolation ---")
        print("-" * 70)

        # Alice asks about Bob's data in Alice's thread
        print("\n[User: Alice] -> Asking about Bob's secret project...")
        res_a3 = app.invoke(
            {"messages": [HumanMessage(content="Do you know what Bob's secret project is?")]},
            config=config_alice
        )
        print(f"AI Response to Alice (Privacy Check):\n{res_a3['messages'][-1].content}\n")

    print("=" * 70)
    print(f"[Info] All multi-user state checkpoints successfully persisted to '{db_path}'.")
    print("=" * 70)

if __name__ == "__main__":
    run_multi_user_demo()
