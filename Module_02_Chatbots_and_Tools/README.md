# Module 02: Building Chatbots & Tool Integration

In this module, you learn how to build conversational AI agents with message state management, tool-calling capabilities, structured Pydantic tools, and error fallbacks using Google Gemini.

---

## 🔑 Key Concepts Covered

1. **`MessagesState`**:
   - Built-in LangGraph state representation holding a list of LangChain message objects (`HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`).
   - Uses `Annotated[list, add_messages]` under the hood to handle message appending and message ID updates.

2. **Tool Binding (`bind_tools`)**:
   - `llm.bind_tools([tool1, tool2])` converts Python tools into JSON schema definitions recognized by Gemini's native function calling interface.

3. **`ToolNode` & `tools_condition`**:
   - **`ToolNode`**: Prebuilt node that extracts `tool_calls` from the latest `AIMessage`, executes the matching function, and returns `ToolMessage` results.
   - **`tools_condition`**: Prebuilt router function that routes to `"tools"` if the last message contains `tool_calls`, or `END` if the LLM produced final text output.

4. **Structured Tools & Error Handling**:
   - Pydantic models for argument validation (`StockQueryInput`).
   - `handle_tool_errors=True` in `ToolNode` catches tool runtime exceptions and feeds error messages back into the LLM context so the agent can recover gracefully.

---

## 📁 Detailed Breakdown of Lessons & Memory Management

### 1. [`01_basic_chatbot.py`](01_basic_chatbot.py) — Ephemeral / Caller-Managed Memory Management

#### 🧠 Memory Management Architecture
- **Memory Pattern Used**: **Caller-Managed / Ephemeral Message History Passing** using built-in `MessagesState`.
- **Under the Hood (`add_messages` Reducer)**:
  `MessagesState` is defined internally by LangGraph as:
  ```python
  class MessagesState(TypedDict):
      messages: Annotated[list[BaseMessage], add_messages]
  ```
  The `add_messages` reducer appends new message objects (`HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`) to the message list while maintaining message ID deduplication.

- **How Multi-Turn Memory Works in `01_basic_chatbot.py`**:
  Because this basic graph **does not attach a server-side checkpointer** (such as `MemorySaver` or `SqliteSaver` introduced in Module 03), graph execution is **stateless across separate `.invoke()` calls**.
  To maintain conversation history across turns, the caller explicitly captures the returned message list from Turn 1 (`state1["messages"]`) and passes it into Turn 2 concatenated with the new user message:
  ```python
  # Turn 1: Stateless invocation
  state1 = graph.invoke({"messages": [HumanMessage(content="Hi! I am learning LangGraph.")]})

  # Turn 2: Caller manually appends new user query to previous history payload
  turn2_input = state1["messages"] + [HumanMessage(content="What is the main benefit...")]
  state2 = graph.invoke({"messages": turn2_input})
  ```

#### ❓ Why is this memory pattern used here?
1. **Foundational Mechanics**: Teaches how `MessagesState` and LangChain message objects work in isolation before adding server-side state persistence.
2. **Stateless API Compatibility**: Mirrors standard REST API microservices where the client UI (React/Next.js) or external database (Redis/Postgres) maintains session history and sends the accumulated payload in the HTTP request body.
3. **Zero Storage Overhead**: Requires no database connections, background storage setup, or checkpointer management.

#### ✅ When SHOULD you use this type of memory?
- **Stateless REST APIs**: When your frontend application or API gateway maintains user chat history and passes the message array with every POST request.
- **Single-Turn / Short Interactions**: One-off tasks like code generation, text summarization, or single-question Q&A agents.
- **Explicit History Filtering**: When the client app needs to filter, prune, or edit messages before sending them to the graph.

#### ❌ When SHOULD you NOT use this type of memory?
- **Long Multi-Turn Production Chatbots**: Transporting massive growing message arrays back and forth over network calls becomes slow, bandwidth-heavy, and error-prone.
- **Human-in-the-Loop & Time Travel**: Ephemeral invocation cannot pause graph execution at breakpoints or rewind state (requires persistent checkpointers like `MemorySaver` / `SqliteSaver`).
- **Multi-User Server Applications**: Lacks thread isolation (`thread_id`) to automatically manage separate user conversations on the backend.
- **Unbounded Context Windows**: Unchecked message list growth will eventually crash the LLM by exceeding token context limits (requires message trimming or summarization reducers).

---

### 2. [`02_tool_calling_agent.py`](02_tool_calling_agent.py) — Autonomous Tool Execution Loop

#### 🎯 Functionality & Key Mechanics
Demonstrates how an agent autonomously detects tool requests, executes function tools, and loops results back using `bind_tools`, `ToolNode`, and `tools_condition`.

#### 🔍 1. Tool Call Detection (`response.tool_calls`)
When Gemini decides to execute a tool, the returned `AIMessage` contains a populated `.tool_calls` list attribute. Inside `agent_node`, we detect tool calls by checking `response.tool_calls`:
```python
if response.tool_calls:
    print(f"   [Tool Call Detected!]: LLM requested {len(response.tool_calls)} tool execution(s):")
    for tc in response.tool_calls:
        print(f"      -> Function: '{tc['name']}' | Arguments: {tc['args']} | Call ID: {tc['id']}")
else:
    print("   [No Tool Call]: LLM produced final textual answer.")
```

#### 🔀 2. The `tools_condition` Router Function
`tools_condition` is a pre-built router from `langgraph.prebuilt`. It inspects the latest message in `state["messages"]`:
- **If `AIMessage.tool_calls` is NOT empty**: Returns string `"tools"` to route execution to `ToolNode`.
- **If `AIMessage.tool_calls` IS empty**: Returns `END` to terminate graph execution as the agent has generated its final text answer.

#### 🔄 3. The Feedback Loop Edge (`builder.add_edge("tools", "agent")`)
```python
builder.add_edge("tools", "agent")
```
This line connects `ToolNode` ("tools") back to `agent_node` ("agent").
- After `ToolNode` executes the function and appends a `ToolMessage` with the result to `state["messages"]`, this edge routes control **back to Gemini**.
- Gemini reads the tool result and decides whether to call another tool (e.g. `add_numbers` after `multiply_numbers`) or synthesize the final textual answer for the user.

---

### 3. [`03_custom_tools_and_fallbacks.py`](03_custom_tools_and_fallbacks.py) — Structured Tools & Exception Recovery

#### 🎯 Functionality
Demonstrates type-safe tool parameters using Pydantic `BaseModel` schemas (`StockQueryInput`) and exception handling.
- `ToolNode(tools, handle_tool_errors=True)` intercepts runtime tool exceptions (e.g. unknown stock ticker) and returns the error message in a `ToolMessage`, allowing Gemini to recover gracefully and notify the user instead of crashing the graph.

---

## 🚀 Running the Lessons

```bash
python Module_02_Chatbots_and_Tools/01_basic_chatbot.py
python Module_02_Chatbots_and_Tools/02_tool_calling_agent.py
python Module_02_Chatbots_and_Tools/03_custom_tools_and_fallbacks.py
```
