import asyncio
import argparse
import time
from src.ingestion.orchestrator import ingest_document
from src.agent.graph import graph_app
from src.config import Config
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class EvaluationScore(BaseModel):
    faithfulness_score: int = Field(description="Score 1-5 on how well the answer is grounded in the retrieved context.")
    relevancy_score: int = Field(description="Score 1-5 on how well the answer addresses the specific question asked.")
    reasoning: str = Field(description="Short explanation for the scores.")

async def evaluate_answer(question: str, answer: str, context: str) -> EvaluationScore:
    """Uses Mistral (Free Tier) as a Judge to evaluate the RAG generation."""
    llm = Config.get_llm(temperature=0.0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert impartial judge evaluating RAG applications. "
                   "Rate the provided Answer based on the Question and the Context.\n"
                   "- Faithfulness (1-5): The answer only contains facts present in the Context.\n"
                   "- Relevancy (1-5): The answer directly answers the Question without waffling."),
        ("human", f"Question: {question}\n\nContext:\n{context}\n\nAnswer to evaluate:\n{answer}")
    ])
    
    chain = prompt | llm.with_structured_output(EvaluationScore)
    return await chain.ainvoke({})

async def run_benchmark():
    print("="*60)
    print(" 🚀 OFFLINE EVALUATION FRAMEWORK (100% Free Tier)")
    print("="*60)
    
    # Ingest the huge dataset
    dataset_file = "huge_dataset.txt"
    print(f"\n[1] Ingesting complex test document: {dataset_file}...")
    await ingest_document(dataset_file)
    print("    ✅ Ingestion successful. Graph & Vector databases populated.")
    
    # Define complex test questions covering structured/unstructured data
    test_cases = [
        {
            "question": "What is the status of the sensor on Mount Everest and how much daily data does AW-001 produce?",
            "expected_focus": "Requires reading the structured markdown table."
        },
        {
            "question": "Who is the CEO of the company that acquired the parent company of OceanData Corp?",
            "expected_focus": "Multi-hop graph traversal across 3 business entities."
        },
        {
            "question": "Project Atlas was initiated by which universities, and who provided the $120 million funding?",
            "expected_focus": "Unstructured text information extraction."
        }
    ]
    
    print("\n[2] Running Multi-Hop RAG Agent & Evaluating Responses...")
    
    total_faithfulness = 0
    total_relevancy = 0
    
    for idx, test in enumerate(test_cases, 1):
        question = test["question"]
        print(f"\n--- Test Case {idx} ---")
        print(f"Q: {question}")
        print(f"Focus: {test['expected_focus']}")
        
        start = time.time()
        # Run graph
        state = {"question": question, "hop_count": 0, "retrieved_context": []}
        result = await graph_app.ainvoke(state)
        latency = time.time() - start
        
        answer = result.get("final_answer", "")
        # Limit context size for judge printing if necessary
        context_str = "\n".join(result.get("retrieved_context", []))
        hop_count = result.get("hop_count", 0)
        
        print(f"\nModel Answer (took {latency:.2f}s, {hop_count} hops):\n=> {answer}")
        
        # Evaluate
        print("\nEvaluating with Mistral LLM-as-a-Judge...")
        score = await evaluate_answer(question, answer, context_str)
        
        print(f"  Faithfulness: {score.faithfulness_score}/5")
        print(f"  Relevancy:    {score.relevancy_score}/5")
        print(f"  Reasoning:    {score.reasoning}")
        
        total_faithfulness += score.faithfulness_score
        total_relevancy += score.relevancy_score
        
    print("\n" + "="*60)
    print(" 📈 FINAL EVALUATION REPORT")
    print("="*60)
    print(f"Average Faithfulness: {total_faithfulness/len(test_cases):.1f} / 5.0")
    print(f"Average Relevancy:    {total_relevancy/len(test_cases):.1f} / 5.0")
    print("Powered exactly by 100% Free-Tier Mistral API and Local Infrastructure.")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
