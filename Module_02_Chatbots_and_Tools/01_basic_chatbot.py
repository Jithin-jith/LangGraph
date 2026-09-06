import os
import warnings
from dotenv import load_dotenv

# Suppress Google GenAI SDK Automatic Function Calling (AFC) recommendation notice
warnings.filterwarnings("ignore", message=".*automatic function calling.*")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END

load_dotenv()

# ============================================================================
# STEP 1: INITIALIZE LLM MODEL
# ============================================================================
# ChatGoogleGenerativeAI communicates directly with Google Gemini 2.5 Flash
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


# ============================================================================
# STEP 2: CHATBOT NODE FUNCTION
# ============================================================================
# MessagesState is a pre-built LangGraph state schema:
#   class MessagesState(TypedDict):
#       messages: Annotated[list[BaseMessage], add_messages]
#
# It automatically handles message list history and message ID updating.
def chatbot_node(state: MessagesState) -> dict:
    """
    Chatbot Node:
    -------------
    Appends a SystemMessage persona prompt, feeds current message history to Gemini,
    and returns the model's AIMessage response to append to state['messages'].
    """
    system_prompt = SystemMessage(
        content="You are a helpful, concise AI tutor specializing in Python and LangGraph state machines."
    )
    # Combine system prompt with full conversation history
    full_prompt = [system_prompt] + state["messages"]
    
    response = llm.invoke(full_prompt)
    return {"messages": [response]}


# ============================================================================
# STEP 3: BUILD AND COMPILE THE GRAPH
# ============================================================================
builder = StateGraph(MessagesState)

# Register chatbot node
builder.add_node("chatbot", chatbot_node)

# Linear flow: START -> chatbot -> END
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()


# ============================================================================
# STEP 4: MULTI-TURN CONVERSATION DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 02 - Lesson 01: Conversational Chatbot with MessagesState ")
    print("=" * 65)

    # Turn 1: User introduces themselves
    print("\n[Turn 1] User: Hi! I am learning LangGraph.")
    turn1_input = [HumanMessage(content="Hi! I am learning LangGraph.")]
    state1 = graph.invoke({"messages": turn1_input})
    print(f"[Turn 1] AI: {state1['messages'][-1].content}\n")

    # Turn 2: User asks follow-up question, passing previous message history
    turn2_query = HumanMessage(content="What is the main benefit of using graphs instead of simple chains?")
    print(f"[Turn 2] User: {turn2_query.content}")
    
    # Pass full accumulated message list from state1 + new user message
    turn2_input = state1["messages"] + [turn2_query]
    state2 = graph.invoke({"messages": turn2_input})
    print(f"[Turn 2] AI: {state2['messages'][-1].content}")
