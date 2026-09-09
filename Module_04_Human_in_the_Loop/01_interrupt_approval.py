"""
Module 04 - Lesson 01: Human Approval Breakpoints (interrupt_before)
====================================================================
Demonstrates how to pause graph execution in LangGraph using `interrupt_before`
for Human-in-the-Loop (HITL) approval before critical or high-risk node actions.

Key Concepts Demonstrated:
---------------------------
1. Checkpointing (`MemorySaver`): Required for state persistence across interrupts.
2. Static Breakpoints (`interrupt_before`): Halts execution immediately before a node.
3. State Inspection (`app.get_state`): Reads current snapshot and pending node (`snapshot.next`).
4. State Modification (`app.update_state`): Injects human decisions into graph state.
5. Execution Resumption (`app.invoke(None, config)`): Continues execution from the saved breakpoint.
"""

import os
import sys
import warnings
import logging
from typing import TypedDict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ============================================================================
# SUPPRESS SDK / LOGGING VERBOSITY
# ============================================================================
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
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ============================================================================
# STEP 1: DEFINE GRAPH STATE
# ============================================================================
# The TransferState dictionary acts as the single source of truth passed 
# between graph nodes. Each node reads from and/or updates this state.
class TransferState(TypedDict):
    sender: str          # Name/ID of account sending funds
    recipient: str       # Name/ID of account receiving funds
    amount: float        # Transfer dollar amount
    approved: bool       # Flag indicating human operator approval
    status_message: str  # Human-readable audit log/status of the transfer

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)


# ============================================================================
# STEP 2: DEFINE GRAPH WORKFLOW NODES
# ============================================================================

def prepare_transfer_node(state: TransferState) -> dict:
    """
    Node 1: Stage Financial Transfer
    -------------------------------
    Prepares transfer metadata and sets the initial status message.
    This node runs automatically when the graph is initially invoked.
    """
    print("\n---> [Node 1: prepare_transfer] Staging Financial Transfer...")
    sender = state.get("sender", "Account_A")
    recipient = state.get("recipient", "Account_B")
    amount = state.get("amount", 0.0)
    return {
        "status_message": f"Transfer of ${amount:.2f} from {sender} to {recipient} staged. Awaiting human approval."
    }


def execute_transfer_node(state: TransferState) -> dict:
    """
    Node 2: Execute Financial Transfer (HIGH RISK ACTION)
    --------------------------------------------------
    Executes financial transfer ONLY after human approval.
    Evaluates state['approved'] to decide whether to complete or reject.
    """
    print("\n---> [Node 2: execute_transfer] Executing Staged Transfer Action!")
    if state.get("approved", False):
        msg = f"SUCCESS: Transferred ${state['amount']:.2f} from {state['sender']} to {state['recipient']}."
    else:
        msg = f"CANCELLED: Transfer of ${state['amount']:.2f} rejected by human reviewer."
    return {"status_message": msg}


# ============================================================================
# STEP 3: BUILD GRAPH WITH INTERRUPT_BEFORE BREAKPOINT
# ============================================================================
# Construct the state graph structure
builder = StateGraph(TransferState)

# Add node functions to the graph builder
builder.add_node("prepare_transfer", prepare_transfer_node)
builder.add_node("execute_transfer", execute_transfer_node)

# Define linear graph flow: START -> prepare_transfer -> execute_transfer -> END
builder.add_edge(START, "prepare_transfer")
builder.add_edge("prepare_transfer", "execute_transfer")
builder.add_edge("execute_transfer", END)

# Checkpointer stores state snapshots in memory.
# MemorySaver is essential for Human-in-the-Loop workflows because graph execution
# is paused and resumed in separate calls, requiring state persistence across invocations.
checkpointer = MemorySaver()

# `interrupt_before=["execute_transfer"]` automatically pauses graph execution
# immediately BEFORE `execute_transfer` executes, persisting current state.
app = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute_transfer"]
)


# ============================================================================
# STEP 4: EXECUTE WORKFLOW WITH HUMAN INTERVENTION
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 04 - Lesson 01: Human-in-the-Loop Breakpoint Approval ")
    print("=" * 65)

    # Each workflow session requires a thread_id in the configurable dictionary
    # so the checkpointer can map state snapshots to specific sessions.
    config = {"configurable": {"thread_id": "transfer-tx-992"}}
    
    initial_state = {
        "sender": "Alice",
        "recipient": "Bob",
        "amount": 5000.0,
        "approved": False
    }

    # ------------------------------------------------------------------------
    # PHASE 1: Launch transfer workflow (Pauses at interrupt_before breakpoint)
    # ------------------------------------------------------------------------
    print("\n--- Phase 1: Launching Transfer Workflow ---")
    app.invoke(initial_state, config=config)

    # Inspect graph state at the breakpoint
    snapshot = app.get_state(config)
    print(f"\n[INTERRUPTED AT BREAKPOINT]")
    print(f"Next node waiting to run: {snapshot.next}")
    print(f"Current State Status   : '{snapshot.values['status_message']}'")
    print(f"Current Approval Status: {snapshot.values.get('approved')}")

    # ------------------------------------------------------------------------
    # PHASE 2: Interactive Human Approval Action (User input y/n)
    # ------------------------------------------------------------------------
    # Prompts the user interactively in the terminal instead of hardcoded/simulated input.
    while True:
        try:
            user_input = input("\n[Human Reviewer Action] Approve transfer of $5000.00 from Alice to Bob? (y/n): ").strip().lower()
            if user_input in ['y', 'n']:
                break
            print("Invalid input. Please enter 'y' for Yes or 'n' for No.")
        except EOFError:
            # Fallback if running in non-interactive environment
            print("\nNon-interactive mode detected. Defaulting to 'y'.")
            user_input = 'y'
            break

    # Process human decision
    if user_input == 'y':
        print("\n--> [Human Action]: APPROVED. Updating graph state with approved=True...")
        # update_state mutates the persisted state snapshot in the checkpointer
        app.update_state(config, {"approved": True})
    else:
        print("\n--> [Human Action]: REJECTED. Updating graph state with approved=False...")
        app.update_state(config, {"approved": False})

    # ------------------------------------------------------------------------
    # PHASE 3: Resume Graph Execution
    # ------------------------------------------------------------------------
    # Passing `None` as the state input signals LangGraph to resume execution
    # from the current saved checkpoint in `config`.
    print("--> Resuming graph execution from breakpoint...")
    final_state = app.invoke(None, config=config)

    # Display final result recorded in state
    print("\n" + "=" * 65)
    print(f"[FINAL WORKFLOW RESULT]: {final_state['status_message']}")
    print("=" * 65)
