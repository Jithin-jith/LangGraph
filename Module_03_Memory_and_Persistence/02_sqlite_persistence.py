"""
Module 03 - Lesson 02: Persistent SQLite Checkpointer
Demonstrates disk-backed state persistence using SqliteSaver for long-term memory across process restarts.
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
    """Conversational node appending response to state messages."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

db_path = "checkpoints.sqlite"


# ============================================================================
# STEP 2: PERSIST STATE TO SQLITE DISK DATABASE
# ============================================================================
# 💡 DISK PERSISTENCE DEMONSTRATION & EXPERIMENT:
# ----------------------------------------------------------------------------
# Run 1: Execute this file as-is. Turn 1 saves "Project Titan" into 'checkpoints.sqlite'.
# Run 2: Comment out Turn 1 (Message 1 below) and run this script a second time!
#        The AI will STILL correctly answer "Project Titan" in Turn 2!
#
# WHY THIS WORKS:
# SqliteSaver reads past state snapshots from the local 'checkpoints.sqlite' database
# file for thread_id="persistent-session-1". Even though Python process memory is
# completely reset when the program exits, the SQLite database on disk holds the
# full conversation history across multiple script executions!
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 03 - Lesson 02: Persistent SQLite Checkpointer ")
    print("=" * 65)

    # Establish SQLite connection context manager
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        # SqliteSaver persists state checkpoints to disk in SQLite tables
        memory = SqliteSaver(conn)
        app = builder.compile(checkpointer=memory)

        # Config defining thread_id session
        config = {"configurable": {"thread_id": "persistent-session-1"}}

        # --------------------------------------------------------------------
        # TURN 1: Saves "Project Titan" into 'checkpoints.sqlite' on disk
        # 🧪 EXPERIMENT: After running the script once, comment out this Turn 1
        # block and re-run the script. Turn 2 below will STILL remember Titan!
        # --------------------------------------------------------------------
        print("\n--- Sending Message 1 ---")
        msg_1 = "Remember my secret project code name is Project Titan."
        print(f"User: {msg_1}")
        r1 = app.invoke(
            {"messages": [HumanMessage(content=msg_1)]},
            config=config
        )
        print(f"AI Response 1: {r1['messages'][-1].content}\n")

        # --------------------------------------------------------------------
        # TURN 2: Reads prior conversation history from 'checkpoints.sqlite'
        # --------------------------------------------------------------------
        print("--- Sending Message 2 ---")
        msg_2 = "What is my secret project code name?"
        print(f"User: {msg_2}")
        r2 = app.invoke(
            {"messages": [HumanMessage(content=msg_2)]},
            config=config
        )
        print(f"AI Response 2: {r2['messages'][-1].content}")

    print(f"\n[Info] Graph state persisted to disk database file: '{db_path}'.")
