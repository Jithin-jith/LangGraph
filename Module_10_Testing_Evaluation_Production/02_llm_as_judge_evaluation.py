"""
Module 10 - Lesson 02: LLM-as-a-Judge Automated Evaluation Benchmark
=====================================================================
Demonstrates evaluating AI agent responses using structured LLM output as an objective judge.
An evaluator LLM benchmarks target agent answers against test datasets and outputs numerical
scores and justifications using a Pydantic schema.

Key Concepts Demonstrated:
---------------------------
1. LLM-as-a-Judge Evaluator Pattern: Using a deterministic LLM (`gemini-2.5-flash` at temperature=0.0)
   to grade answer quality against expected topics.
2. Structured Evaluation Schema (`EvalScore`): Using Pydantic to enforce structured output
   (`relevance_score`, `completeness_score`, `reasoning`).
3. Automated Evaluation Harness: Iterating over benchmark test cases to generate quantitative
   quality metrics across target agent deployments.
"""

import sys
import os
import warnings
import logging
from typing import TypedDict, List
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
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ============================================================================
# STEP 1: DEFINE STRUCTURED EVALUATION SCORE SCHEMA
# ============================================================================
# Pydantic schema enforcing structured judge outputs
class EvalScore(BaseModel):
    relevance_score: int = Field(description="Score from 1 to 10 for answer relevance")
    completeness_score: int = Field(description="Score from 1 to 10 for answer completeness")
    reasoning: str = Field(description="Brief justification for given scores")

evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
structured_evaluator = evaluator_llm.with_structured_output(EvalScore)


# ============================================================================
# STEP 2: DEFINE TARGET AGENT GRAPH BEING EVALUATED
# ============================================================================
class TargetState(TypedDict):
    question: str
    answer: str

target_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

def answer_node(state: TargetState) -> dict:
    """Target agent node answering question."""
    res = target_llm.invoke(f"Answer concisely: {state['question']}").content.strip()
    return {"answer": res}

builder = StateGraph(TargetState)
builder.add_node("answer_node", answer_node)
builder.add_edge(START, "answer_node")
builder.add_edge("answer_node", END)

agent_app = builder.compile()


# ============================================================================
# STEP 3: BENCHMARK DATASET & AUTOMATED EVALUATION SUITE
# ============================================================================
dataset = [
    {
        "question": "Explain what a Graph State is in LangGraph.",
        "expected_topics": ["TypedDict", "schema", "nodes", "shared state"]
    },
    {
        "question": "What is the difference between add_edge and add_conditional_edges?",
        "expected_topics": ["static routing", "dynamic routing", "router function"]
    }
]

# ============================================================================
# STEP 4: EXECUTION DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Module 10 - Lesson 02: LLM-as-a-Judge Automated Evaluation Benchmark ")
    print("=" * 65)

    for idx, test_case in enumerate(dataset, 1):
        q = test_case["question"]
        print(f"\n[Eval Test Case {idx}]: Query: '{q}'")
        
        # 1. Execute agent graph to get actual response
        output = agent_app.invoke({"question": q})
        answer = output["answer"]
        print(f"Agent Response:\n{answer}\n")

        # 2. Invoke LLM-as-a-Judge Evaluator
        eval_prompt = (
            f"Evaluate the AI Agent response to the question.\n"
            f"Question: {q}\n"
            f"Expected key topics: {test_case['expected_topics']}\n"
            f"Actual Agent Response: {answer}"
        )
        score: EvalScore = structured_evaluator.invoke(eval_prompt)
        
        print(f"--> [JUDGE SCORES]: Relevance = {score.relevance_score}/10 | Completeness = {score.completeness_score}/10")
        print(f"--> [JUDGE REASONING]: {score.reasoning}")
        print("=" * 65)
