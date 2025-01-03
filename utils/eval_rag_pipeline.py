# eval_synthetic_qa_weave.py

import os
import json
import asyncio

import weave
from weave import Evaluation

import Levenshtein  # pip install python-Levenshtein

from retrieval_graph import graph  # Your retrieval pipeline
from langchain_core.runnables import RunnableConfig
from retrieval_graph.retrieval import make_pinecone_retriever

def load_data(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)

@weave.op()
async def evaluate_retrieval_only(question: str, expected: str) -> dict:
    """Evaluate if the expected answer appears in the top-K retrieved documents."""
    try:
        # Get Pinecone retriever
        retriever = await make_pinecone_retriever(
            user_id="eval_user",  # Using a fixed user ID for evaluation
            k=5  # Top-K documents to retrieve
        )
        
        # Retrieve documents
        docs = await retriever.aget_relevant_documents(question)
        
        # Check if expected answer appears in any retrieved document
        expected_text = expected.strip().lower()
        found = False
        best_score = 0
        
        for doc in docs:
            doc_text = doc.page_content.strip().lower()
            # Check for exact containment
            if expected_text in doc_text:
                found = True
                best_score = 1.0
                break
            # Fallback to Levenshtein similarity for fuzzy matching
            score = 1 - (Levenshtein.distance(expected_text, doc_text) / max(len(expected_text), len(doc_text)))
            best_score = max(best_score, score)
            if score > 0.8:  # High similarity threshold
                found = True
                break
        
        return {
            "retrieval_match": found,
            "retrieval_score": best_score,
            "num_docs": len(docs)
        }
        
    except Exception as e:
        print(f"Error in retrieval evaluation: {str(e)}")
        return {
            "retrieval_match": False,
            "retrieval_score": 0.0,
            "num_docs": 0
        }

@weave.op()
def retrieval_scorer(expected: str, model_output: dict) -> dict:
    """Score the retrieval results."""
    try:
        return {
            "found_doc": model_output.get("retrieval_match", False),
            "retrieval_score": model_output.get("retrieval_score", 0.0),
            "num_docs": model_output.get("num_docs", 0)
        }
    except Exception as e:
        print(f"Error in retrieval scoring: {str(e)}")
        return {
            "found_doc": False,
            "retrieval_score": 0.0,
            "num_docs": 0
        }

async def main():
    weave.init("mirror-agent-evals")

    data = load_data("data/synthetic_qa_dataset.json")
    subset_data = data[:5]  # Testing with first 5 examples

    examples = []
    for item in subset_data:
        examples.append({
            "question": item["question"],
            "expected": item["answer"]
        })

    # Create retrieval-only evaluation
    retrieval_evaluation = Evaluation(
        dataset=examples,
        scorers=[retrieval_scorer],
        name="Retrieval-only-check"
    )

    print("\nRunning retrieval-only evaluation...")
    await retrieval_evaluation.evaluate(
        evaluate_retrieval_only,
        __weave={"display_name": "Retrieval Pipeline Baseline"}
    )

    print("Retrieval evaluation complete. View results in Weave UI.")

if __name__ == "__main__":
    asyncio.run(main())