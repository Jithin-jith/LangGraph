import os
import sys
import warnings
import sqlite3
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Suppress Google GenAI SDK Automatic Function Calling (AFC) recommendation notice
warnings.filterwarnings("ignore", message=".*automatic function calling.*")

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
        print("\nSending Message 1...")
        r1 = app.invoke(
            {"messages": [HumanMessage(content="Remember my secret project code name is Project Titan.")]},
            config=config
        )
        print(f"AI Response 1: {r1['messages'][-1].content}\n")

        # --------------------------------------------------------------------
        # TURN 2: Reads prior conversation history from 'checkpoints.sqlite'
        # --------------------------------------------------------------------
        print("Sending Message 2...")
        r2 = app.invoke(
            {"messages": [HumanMessage(content="What is my secret project code name?")]},
            config=config
        )
        print(f"AI Response 2: {r2['messages'][-1].content}")

    print(f"\n[Info] Graph state persisted to disk database file: '{db_path}'.")
