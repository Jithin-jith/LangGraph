"""
Module 04 - Lesson 02: Human State Editing & Resuming
====================================================================
Demonstrates how a human operator can inspect, modify, and update state values
while a LangGraph workflow is paused at an `interrupt_before` breakpoint.

Key Concepts Demonstrated:
--------------------------
1. AI Drafting Node (`draft_email_node`): Uses Gemini LLM to generate initial content.
2. Static Breakpoint (`interrupt_before=["send_email"]`): Pauses execution after drafting
   but BEFORE executing a critical external action (sending email).
3. State Inspection (`app.get_state`): Allows human operators to review pending outputs.
4. State Modification (`app.update_state`): The core LangGraph API for programmatically
   updating state snapshot values saved in the checkpointer before resumption.
5. Execution Resumption (`app.invoke(None, config)`): Continues graph execution from the
   saved breakpoint using the newly updated state.
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
# EmailState schema maintains information throughout the lifecycle of the email draft.
class EmailState(TypedDict):
    recipient_email: str  # Destination email address
    topic: str            # Outreach topic prompt provided to the LLM
    draft_subject: str    # Generated subject line (subject to human editing)
    draft_body: str       # Generated email body (subject to human editing)
    final_status: str     # Audit status after sending

# Initialize Gemini LLM for creative text generation
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


# ============================================================================
# STEP 2: DEFINE GRAPH WORKFLOW NODES
# ============================================================================

def draft_email_node(state: EmailState) -> dict:
    """
    Node 1: Draft Email Node
    -----------------------
    Uses Gemini LLM to generate an initial draft cold email based on state['topic'].
    Returns draft_subject and draft_body to update graph state.
    """
    print("\n---> [Node 1: draft_email] Generating AI Draft Email with Gemini...")
    topic = state["topic"]
    prompt = f"Draft a professional cold outreach email about: {topic}. Provide subject line on line 1, body on remaining lines."
    res = llm.invoke(prompt).content
    
    lines = res.strip().split("\n", 1)
    subject = lines[0].replace("Subject:", "").strip()
    body = lines[1].strip() if len(lines) > 1 else lines[0]
    
    return {
        "draft_subject": subject,
        "draft_body": body
    }


def send_email_node(state: EmailState) -> dict:
    """
    Node 2: Send Email Node (CRITICAL / EXTERNAL ACTION)
    ---------------------------------------------------
    Sends the final email using the current values in state['draft_subject']
    and state['draft_body']. This node ONLY runs after human review/edits.
    """
    print("\n---> [Node 2: send_email] Dispatching Final Approved Email...")
    recipient = state["recipient_email"]
    subject = state["draft_subject"]
    body = state["draft_body"]
    
    print("\n" + "=" * 55)
    print(f" TO     : {recipient}")
    print(f" SUBJECT: {subject}")
    print("-" * 55)
    print(f" BODY   :\n{body}")
    print("=" * 55)
    
    return {"final_status": f"Email successfully dispatched to {recipient}."}


# ============================================================================
# STEP 3: BUILD GRAPH WITH INTERRUPT_BEFORE BREAKPOINT
# ============================================================================
builder = StateGraph(EmailState)

# Add node functions to state graph builder
builder.add_node("draft_email", draft_email_node)
builder.add_node("send_email", send_email_node)

# Define sequential edges: START -> draft_email -> send_email -> END
builder.add_edge(START, "draft_email")
builder.add_edge("draft_email", "send_email")
builder.add_edge("send_email", END)

# MemorySaver checkpointer retains graph snapshots in memory across interrupts
checkpointer = MemorySaver()

# `interrupt_before=["send_email"]` halts execution after draft_email completes
# and right before send_email executes, allowing human review and state editing.
app = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["send_email"]
)


# ============================================================================
# STEP 4: HUMAN REVIEW AND STATE EDITING DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 04 - Lesson 02: Human Editing of Interrupted State ")
    print("=" * 65)

    # Session configuration containing thread_id key for checkpointing
    config = {"configurable": {"thread_id": "email-task-404"}}
    
    # ------------------------------------------------------------------------
    # PHASE 1: Run AI Drafting Node (Pauses before send_email)
    # ------------------------------------------------------------------------
    print("\n--- Phase 1: Launching AI Email Generator ---")
    app.invoke({
        "recipient_email": "client@example.com",
        "topic": "Partnership proposal for AI consultancy services"
    }, config=config)

    # ------------------------------------------------------------------------
    # PHASE 2: Inspect Interrupted State & Perform Human Edits
    # ------------------------------------------------------------------------
    snapshot = app.get_state(config)
    print(f"\n[INTERRUPTED AT BREAKPOINT]")
    print(f"Next Node Waiting: {snapshot.next}")
    print(f"\n[Generated Subject]: {snapshot.values['draft_subject']}")
    print(f"[Generated Body Snippet]: {snapshot.values['draft_body'][:150]}...\n")

    print("--- Human Reviewer Modifying Draft State ---")
    edited_subject = "[URGENT] Proposal: AI Infrastructure & Consultancy Services"
    edited_body = snapshot.values['draft_body'] + "\n\nPS: Let us know your availability for a 15-minute intro call this Thursday."

    # `app.update_state()` overwrites the saved checkpoint values in memory.
    # When the graph resumes, send_email will receive these modified values.
    app.update_state(config, {
        "draft_subject": edited_subject,
        "draft_body": edited_body
    })

    print("--> State snapshot successfully updated with human edits.")

    # ------------------------------------------------------------------------
    # PHASE 3: Resume Graph Execution
    # ------------------------------------------------------------------------
    # Calling app.invoke(None, config) resumes execution from the breakpoint
    print("--> Resuming send_email node with updated state...")
    res = app.invoke(None, config=config)

    # Display final result returned by the workflow
    print(f"\n[FINAL STATUS]: {res['final_status']}")
