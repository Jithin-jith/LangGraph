import os
import warnings
from dotenv import load_dotenv

# Suppress Google GenAI SDK Automatic Function Calling (AFC) recommendation notice
warnings.filterwarnings("ignore", message=".*automatic function calling.*")

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ============================================================================
# STEP 1: DEFINE TOOLS
# ============================================================================
# Use `@tool` decorator to convert standard Python functions into LangChain Tools.
# Type annotations and docstrings serve as the schema description for Gemini!

@tool
def add_numbers(a: float, b: float) -> float:
    """Adds two numbers together and returns the sum."""
    print(f"   [Tool Executed]: add_numbers({a}, {b})")
    return a + b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiplies two numbers together and returns the product."""
    print(f"   [Tool Executed]: multiply_numbers({a}, {b})")
    return a * b

# Tool registry list
tools = [add_numbers, multiply_numbers]

# Bind tools to Gemini model. This converts Python tools into JSON Schema function signatures.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
llm_with_tools = llm.bind_tools(tools)


# ============================================================================
# STEP 2: DEFINE AGENT NODE
# ============================================================================
def agent_node(state: MessagesState) -> dict:
    """
    Agent Node:
    -----------
    Invokes Gemini with tool definitions. Gemini can either:
    1. Decide to call one or more tools (returns AIMessage with `tool_calls`).
    2. Provide a final textual response (returns standard AIMessage).
    """
    print("\n---> [Agent Node] Calling Gemini Model with Tools...")
    response = llm_with_tools.invoke(state["messages"])
    
    # ------------------------------------------------------------------------
    # HOW TO DETECT IF THE LAST MESSAGE CONTAINED A TOOL CALL:
    # ------------------------------------------------------------------------
    # An AIMessage returned by a tool-enabled LLM contains a `.tool_calls` list attribute.
    # If `response.tool_calls` is non-empty, the LLM has requested one or more tool executions!
    if response.tool_calls:
        print(f"   [Tool Call Detected!]: LLM requested {len(response.tool_calls)} tool execution(s):")
        for tc in response.tool_calls:
            print(f"      -> Function: '{tc['name']}' | Arguments: {tc['args']} | Call ID: {tc['id']}")
    else:
        print("   [No Tool Call]: LLM produced final textual answer.")

    return {"messages": [response]}


# ============================================================================
# STEP 3: BUILD GRAPH WITH PREBUILT TOOLNODE & TOOLS_CONDITION
# ============================================================================
builder = StateGraph(MessagesState)

# Register Agent Node
builder.add_node("agent", agent_node)

# ToolNode is a pre-built LangGraph node that automatically executes tool calls
# returned by the agent and formats results into ToolMessage objects.
builder.add_node("tools", ToolNode(tools))

# Entrypoint: START -> agent
builder.add_edge(START, "agent")

# ----------------------------------------------------------------------------
# EXPLANATION: `tools_condition`
# ----------------------------------------------------------------------------
# `tools_condition` is a pre-built router function from langgraph.prebuilt.
# It inspects the latest message in state["messages"] (which is the AIMessage):
#   - If AIMessage.tool_calls is non-empty -> returns string "tools" (routing to ToolNode)
#   - If AIMessage.tool_calls is empty     -> returns END (stopping graph execution)
builder.add_conditional_edges("agent", tools_condition)

# ----------------------------------------------------------------------------
# EXPLANATION: `builder.add_edge("tools", "agent")`
# ----------------------------------------------------------------------------
# This creates a FEEDBACK LOOP connecting ToolNode ("tools") back to Agent Node ("agent").
# After ToolNode executes the requested function and appends a ToolMessage with the result,
# this edge routes back to the agent so Gemini can inspect the tool output and decide:
#   a) Call another tool (e.g. add 150 after multiplying 45 * 12).
#   b) Synthesize the tool result into a final textual answer for the user.
builder.add_edge("tools", "agent")

graph = builder.compile()


# ============================================================================
# STEP 4: EXECUTION DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 02 - Lesson 02: Tool-Calling Agent with ToolNode & tools_condition ")
    print("=" * 65)

    user_prompt = "What is 45 multiplied by 12, and then add 150 to the result?"
    print(f"User Query: {user_prompt}")

    # Run agent graph
    result = graph.invoke(input={"messages": [HumanMessage(content=user_prompt)]})

    print("\n" + "=" * 65)
    print("--- Final Agent Response ---")
    print("=" * 65)
    print(result["messages"][-1].content[0]['text'])
