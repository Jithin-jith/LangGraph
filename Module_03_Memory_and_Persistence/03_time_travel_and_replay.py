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
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ============================================================================
# STEP 1: INITIALIZE GRAPH WITH CHECKPOINTER
# ============================================================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

def chatbot_node(state: MessagesState) -> dict:
    """Chatbot node generating conversational response."""
    res = llm.invoke(state["messages"])
    return {"messages": [res]}

builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Checkpointers store state history snapshots required for state inspection and time travel
checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)


# ============================================================================
# STEP 2: DEMONSTRATE TIME TRAVEL, SNAPSHOT INSPECTION & STATE EDITING
# ============================================================================
# 💡 WHAT IS TIME TRAVEL & STATE EDITING?
# ----------------------------------------------------------------------------
# In standard LLM chat applications, past messages cannot be easily modified once sent.
# In LangGraph, checkpointers maintain a timeline of immutable state snapshots.
# Using state inspection and editing methods, developers can:
#   1. Inspect graph state at any point using `app.get_state(config)`.
#   2. Traverse full checkpoint execution history using `app.get_state_history(config)`.
#   3. Fork, edit, or inject state updates using `app.update_state(config, values)`.
#   4. Replay or resume execution from the modified state using `app.invoke(None, config)`.
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 03 - Lesson 03: Time Travel, State Inspection & Editing ")
    print("=" * 65)

    config = {"configurable": {"thread_id": "time-travel-demo"}}

    # Turn 1: User introduces buying intent
    print("\n--- Turn 1: Initial User Query ---")
    user_msg_1 = "I want to buy a laptop."
    print(f"User: {user_msg_1}")
    r1 = app.invoke({"messages": [HumanMessage(content=user_msg_1)]}, config=config)
    print(f"AI: {r1['messages'][-1].content}\n")

    # Turn 2: User sets budget to $1000
    print("--- Turn 2: Initial Budget Statement ---")
    user_msg_2 = "My budget is $1000."
    print(f"User: {user_msg_2}")
    r2 = app.invoke({"messages": [HumanMessage(content=user_msg_2)]}, config=config)
    print(f"AI: {r2['messages'][-1].content}\n")

    # ------------------------------------------------------------------------
    # 1. INSPECT CURRENT STATE SNAPSHOT (`app.get_state`)
    # ------------------------------------------------------------------------
    # `app.get_state(config)` returns a `StateSnapshot` object containing:
    #   - `.values`: Current state channels (e.g. `messages` list)
    #   - `.next`: Tuple of next nodes to execute (empty tuple `()` if finished)
    #   - `.config`: Configuration containing thread_id and active checkpoint_id
    # ------------------------------------------------------------------------
    current_state = app.get_state(config)
    print("=" * 65)
    print("[1. Current State Snapshot Inspection]")
    print(f"   - Active Checkpoint ID: {current_state.config['configurable']['checkpoint_id']}")
    print(f"   - Total Messages in State: {len(current_state.values['messages'])}")
    print(f"   - Next Node to Run: {current_state.next}")

    # ------------------------------------------------------------------------
    # 2. BROWSE CHECKPOINT HISTORY (`app.get_state_history`)
    # ------------------------------------------------------------------------
    # `app.get_state_history(config)` returns a generator yielding past state
    # snapshots in reverse chronological order (newest to oldest).
    # ------------------------------------------------------------------------
    print("\n[2. Checkpoint Execution History Traversal]")
    history = list(app.get_state_history(config))
    for idx, snapshot in enumerate(history):
        checkpoint_id = snapshot.config['configurable']['checkpoint_id']
        num_msgs = len(snapshot.values.get('messages', []))
        print(f"   Checkpoint [{idx}]: ID={checkpoint_id} | Messages={num_msgs} | Next={snapshot.next}")

    # ------------------------------------------------------------------------
    # 3. TIME TRAVEL / STATE EDITING (`app.update_state`)
    # ------------------------------------------------------------------------
    # `app.update_state(config, values)` injects new state updates into the thread.
    # Because state["messages"] uses the `add_messages` reducer, passing a new
    # HumanMessage appends it to history and generates a fresh checkpoint snapshot.
    # Here we correct the user's budget from $1000 to $2500 for a gaming laptop.
    # ------------------------------------------------------------------------
    print("\n[3. Time Travel: Injecting Budget Correction State Update]")
    correction_msg = "Correction: My budget is actually $2500 for a high-end gaming laptop."
    print(f"User (State Correction): {correction_msg}")
    app.update_state(
        config,
        {"messages": [HumanMessage(content=correction_msg)]}
    )
    print("   -> State updated successfully! New checkpoint created.")

    # ------------------------------------------------------------------------
    # 4. RESUME GRAPH EXECUTION WITH MODIFIED STATE (`app.invoke(None, config)`)
    # ------------------------------------------------------------------------
    # Calling `app.invoke(None, config)` executes the graph from the newest
    # updated checkpoint. We pass `None` as the input payload because the
    # updated state is already stored in the checkpointer!
    # ------------------------------------------------------------------------
    print("\n[4. Resuming Graph Execution with Modified State Snapshot]")
    res = app.invoke(None, config=config)
    print(f"AI (Post-Time Travel Correction):\n{res['messages'][-1].content}")
    print("=" * 65)
