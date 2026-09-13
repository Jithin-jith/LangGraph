"""
Module 10 - Lesson 03: Production FastAPI REST Service Deployment
=====================================================================
Demonstrates deploying a compiled LangGraph agent as a production-grade FastAPI REST API.
Exposes a `/chat` endpoint supporting per-thread conversational state persistence using `MemorySaver`.

Key Concepts Demonstrated:
---------------------------
1. FastAPI REST API Wrapper: Wrapping a compiled `StateGraph` inside a FastAPI web server.
2. Per-Thread State Isolation: Passing `thread_id` in API requests so `MemorySaver` isolates
   independent user sessions across API clients.
3. Health Check Endpoints: Exposing `/health` for Kubernetes / load balancer liveness probes.
4. Input Validation Models: Using Pydantic `ChatRequest` and `ChatResponse` schemas to validate
   REST payloads.
"""

import sys
import os
import warnings
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

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
        if self.original_stderr:
            self.original_stderr.write(msg)

    def flush(self):
        if self.original_stderr and hasattr(self.original_stderr, "flush"):
            self.original_stderr.flush()

sys.stderr = StderrFilter(sys.stderr)

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ============================================================================
# STEP 1: INITIALIZE LANGGRAPH AGENT & CHECKPOINTER
# ============================================================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

def chatbot_node(state: MessagesState) -> dict:
    """Conversational node generating AI response."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

checkpointer = MemorySaver()
agent_app = builder.compile(checkpointer=checkpointer)


# ============================================================================
# STEP 2: BUILD FASTAPI REST SERVICE
# ============================================================================
app = FastAPI(
    title="LangGraph Gemini Agent API",
    description="Production REST API Service for LangGraph Conversational Agent",
    version="1.0.0"
)

# REST Request/Response Models
class ChatRequest(BaseModel):
    thread_id: str = Field(..., description="Unique conversation session identifier")
    message: str = Field(..., description="User prompt text")

class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    message_count: int


# Health Check Endpoint for Kubernetes / Load Balancers
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "LangGraph Production API"}


# Chat Endpoint with Thread State Persistence
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    POST /chat:
    -----------
    Accepts thread_id and message. Uses checkpointer to automatically resume
    and persist conversation history per thread_id.
    """
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        state_result = agent_app.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )
        messages = state_result["messages"]
        reply_text = messages[-1].content
        
        return ChatResponse(
            thread_id=request.thread_id,
            reply=reply_text,
            message_count=len(messages)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STEP 3: RUN SERVICE WITH UVICORN
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 65)
    print(" Module 10 - Lesson 03: Production FastAPI REST Service Deployment ")
    print("=" * 65)
    print("Starting FastAPI server at http://127.0.0.1:8000 ... (Press Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
