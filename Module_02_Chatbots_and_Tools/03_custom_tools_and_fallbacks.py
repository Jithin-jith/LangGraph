"""
Module 02 - Lesson 03: Structured Custom Tools & Exception Fallbacks
Demonstrates Pydantic schema validation for custom tools and handling tool runtime errors using handle_tool_errors.
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
from typing import Type
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ============================================================================
# STEP 1: PYDANTIC TOOL SCHEMA & CUSTOM TOOL DEFINITION
# ============================================================================
# Pydantic schema for strict input argument validation
class StockQueryInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol in uppercase (e.g. AAPL, GOOGL, NVDA)")
    period: str = Field(default="1d", description="Time horizon quote: 1d, 1w, 1m, 1y")

@tool(args_schema=StockQueryInput)
def get_stock_price(ticker: str, period: str = "1d") -> str:
    """Fetches real-time stock quotes for a given ticker symbol."""
    symbol = ticker.upper()
    mock_db = {"AAPL": "$230.50", "GOOGL": "$175.20", "NVDA": "$125.80"}
    
    if symbol in mock_db:
        return f"Ticker {symbol} price ({period}): {mock_db[symbol]}"
    else:
        # Intentionally raise an error for unknown tickers to demonstrate tool error recovery!
        raise ValueError(f"Ticker symbol '{symbol}' was not found in active stock directory.")


# ============================================================================
# STEP 2: TOOL NODE WITH BUILT-IN EXCEPTION HANDLING
# ============================================================================
tools = [get_stock_price]

# `handle_tool_errors=True` catches tool exceptions and feeds error messages
# back to the LLM inside a ToolMessage, allowing the LLM to recover gracefully.
tool_node = ToolNode(tools, handle_tool_errors=True)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
llm_with_tools = llm.bind_tools(tools)


# ============================================================================
# STEP 3: BUILD THE AGENT GRAPH
# ============================================================================
def agent_node(state: MessagesState) -> dict:
    """Agent node invoking Gemini with stock query tool."""
    print("\n---> [Agent Node] Processing User Query...")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()


# ============================================================================
# STEP 4: EXECUTION DEMO (SUCCESS vs ERROR FALLBACK)
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 02 - Lesson 03: Structured Tools & Fallback Handling ")
    print("=" * 65)

    # Test 1: Valid Ticker Symbol (AAPL) -> Tool Success
    print("\n--- Test 1: Valid Ticker Query (AAPL) ---")
    res1 = graph.invoke({"messages": [HumanMessage(content="What is the current stock price of AAPL?")]})
    print(f"Final Response: {res1['messages'][-1].content}")

    # Test 2: Invalid Ticker Symbol (TSLA) -> Tool Error & Graceful Recovery
    print("\n--- Test 2: Invalid Ticker Query (TSLA - Triggers Error Fallback) ---")
    res2 = graph.invoke({"messages": [HumanMessage(content="Check stock price for TSLA.")]})
    print(f"Final Response: {res2['messages'][-1].content[0]['text']}")
