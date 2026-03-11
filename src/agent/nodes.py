import os
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.agent.state import AgentState
from src.config import Config

class RouteDecision(BaseModel):
    is_complex: bool = Field(description="True if the question requires synthesizing multiple pieces of information or multi-hop reasoning. False if simple.")

class SubQueries(BaseModel):
    queries: list[str] = Field(description="List of isolated, semantic sub-queries.")

class EvaluationDecision(BaseModel):
    is_sufficient: bool = Field(description="True if the context fully answers the question, False otherwise.")

def get_llm():
    return Config.get_llm(temperature=0.0)

async def router_node(state: AgentState) -> Dict[str, Any]:
    """Classifies if the query needs multi-hop decomposition."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert query analyzer. Determine if the user's question requires multi-hop reasoning or is simple."),
        ("human", "{question}")
    ])
    chain = prompt | llm.with_structured_output(RouteDecision)
    decision = await chain.ainvoke({"question": state["question"]})
    return {"is_complex": decision.is_complex, "hop_count": 0}

async def decomposer_node(state: AgentState) -> Dict[str, Any]:
    """Breaks down a complex query into simpler parallel sub-queries."""
    if not state.get("is_complex"):
        return {"sub_queries": [state["question"]]}
        
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Break this complex question down into 2-3 isolated sub-questions that can be searched independently."),
        ("human", "{question}")
    ])
    chain = prompt | llm.with_structured_output(SubQueries)
    result = await chain.ainvoke({"question": state["question"]})
    return {"sub_queries": result.queries}

async def evaluator_node(state: AgentState) -> Dict[str, Any]:
    """Evaluates if the currently retrieved context is enough to answer the original question."""
    llm = get_llm()
    context_str = "\n".join(state.get("retrieved_context", []))
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Based on the context, can we fully answer the user's question? If Yes, set is_sufficient to True."),
        ("human", "Question: {question}\n\nContext: {context}")
    ])
    chain = prompt | llm.with_structured_output(EvaluationDecision)
    result = await chain.ainvoke({"question": state["question"], "context": context_str})
    
    # We update hop_count here as it represents one full retrieval cycle evaluation
    new_hop_count = state.get("hop_count", 0) + 1
    # We return the flag just to be used by the conditional edge routing, we don't strictly need to save it to state
    # but returning it updates the state if we added it, but let's just use it in the edge
    return {"hop_count": new_hop_count, "is_sufficient": result.is_sufficient}

async def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Synthesizes the final answer using all retrieved contexts."""
    llm = get_llm()
    context_str = "\n\n".join(state.get("retrieved_context", []))
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI answering questions based on the provided context. Use inline citations (e.g. [doc_id: chunk_id]) if possible."),
        ("human", "Question: {question}\n\nContext: {context}")
    ])
    chain = prompt | llm
    result = await chain.ainvoke({"question": state["question"], "context": context_str})
    return {"final_answer": result.content}
