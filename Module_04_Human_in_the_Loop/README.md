# Module 04: Human-in-the-Loop (HITL) & Control Flow

Human-in-the-loop (HITL) workflows allow human operators to inspect, approve, reject, or edit graph state before dangerous, critical, or high-stakes nodes execute.

---

## 🔑 Key Concepts Covered

1. **`interrupt_before` Breakpoints**:
   - Registered via `builder.compile(checkpointer=memory, interrupt_before=["node_name"])`.
   - Halts execution immediately prior to running `node_name`, automatically persisting the current state snapshot to the checkpointer under a session `thread_id`.

2. **State Inspection & Pending Node Detection**:
   - `app.get_state(config)`: Retrieves the current checkpoint snapshot.
   - `snapshot.next`: Indicates which node is paused and waiting to execute next (e.g., `('execute_transfer',)` or `('send_email',)`).
   - `snapshot.values`: Exposes all current graph state dictionary keys and values.

3. **Human State Injection (`update_state`)**:
   - `app.update_state(config, {"approved": True, "draft_body": "Updated text"})`: Overwrites or merges state snapshot values stored in the checkpointer at the breakpoint.

4. **Execution Resumption**:
   - `app.invoke(None, config=config)`: Passing `None` signals LangGraph to resume workflow execution directly from the interrupted checkpoint node using updated state values.

---

## 📁 Lesson Scripts Breakdown

### 1. [`01_interrupt_approval.py`](01_interrupt_approval.py): Human Approval Breakpoint (Binary Decision)
- **Goal**: Pause financial transfers before execution so a human operator can review and explicitly approve (`y`) or reject (`n`) the transaction.
- **Workflow**:
  - **Node 1 (`prepare_transfer`)**: Stages the dollar amount, sender, and recipient.
  - **Breakpoint (`interrupt_before=["execute_transfer"]`)**: Halts execution before money is moved.
  - **Human Interaction**: Prompts the user interactively (`y`/`n`) in the terminal.
  - **State Update**: Updates `approved=True` or `approved=False` via `app.update_state()`.
  - **Node 2 (`execute_transfer`)**: Evaluates `approved` flag to complete or cancel the transaction upon workflow resumption.

### 2. [`02_state_modification_hitl.py`](02_state_modification_hitl.py): Human State Editing & Resuming (Content Modification)
- **Goal**: Allow a human reviewer to inspect, edit, and enhance AI-generated cold outreach emails before dispatching them to recipients.
- **Workflow**:
  - **Node 1 (`draft_email`)**: Uses Gemini LLM (`gemini-2.5-flash`) to generate a draft email subject and body based on a specified topic.
  - **Breakpoint (`interrupt_before=["send_email"]`)**: Halts execution after AI generation and before sending out the email.
  - **Human Review & Editing**: Inspects `snapshot.values['draft_subject']` and `snapshot.values['draft_body']`, then modifies subject line and appends custom text.
  - **State Update**: Calls `app.update_state(config, {"draft_subject": ..., "draft_body": ...})` to persist human edits into the checkpointer.
  - **Node 2 (`send_email`)**: Executes with the human-edited content upon calling `app.invoke(None, config=config)`.

---

## 🚀 Running the Lessons

Run each script from the terminal to experience the interactive human-in-the-loop workflows:

```bash
# Test human approval / rejection breakpoint with interactive terminal input:
python 01_interrupt_approval.py

# Test human state editing of AI-generated email drafts:
python 02_state_modification_hitl.py
```
