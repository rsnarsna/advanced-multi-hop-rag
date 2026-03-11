"""
Offline Ragas Evaluation Benchmark script.
"""
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

def run_benchmark():
    # Synthetic test dataset 
    data = {
        "question": ["Who is the CEO of the company that acquired StartupY?"],
        "contexts": [["StartupY was acquired by CompanyX.", "CompanyX CEO is Alice."]],
        "ground_truth": ["Alice is the CEO of CompanyX."],
        "answer": ["The CEO of the company that acquired StartupY is Alice."]
    }
    
    dataset = Dataset.from_dict(data)
    
    # Evaluate metrics against the synthetic generated contexts
    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
        ],
    )
    
    print("Ragas Evaluation Results:")
    print(result)

if __name__ == "__main__":
    run_benchmark()
