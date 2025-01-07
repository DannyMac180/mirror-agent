# eval_synthetic_qa_weave.py

import os
import json
import asyncio

import weave
from weave import Evaluation

import Levenshtein  # pip install python-Levenshtein

from langchain_core.runnables import RunnableConfig
from retrieval_graph.configuration import IndexConfiguration
from retrieval_graph import graph  # Your retrieval pipeline
from retrieval_graph.retrieval import make_retriever

def load_data(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)

@weave.op()
async def evaluate_retrieval_only(question: str, expected: str) -> dict:
    """Evaluate if the expected answer appears in the top-K retrieved documents."""
    try:
        # Construct a config that matches how your docs were indexed
        config = RunnableConfig(
            configurable={
                "user_id": "1",
                "retriever_provider": "pinecone",
                "embedding_model": "openai/text-embedding-3-small",
                "search_kwargs": {"k": 5}
            }
        )

        # Use the shared make_retriever(...) which
        # calls make_pinecone_retriever internally.
        with make_retriever(config) as retriever:
            docs = await retriever.ainvoke(question, config)
        
        # Check if expected answer appears in any retrieved document
        expected_text = expected.strip().lower()
        found = False
        best_score = 0
        
        # Prepare document previews
        doc_previews = []
        for doc in docs:
            doc_text = doc.page_content.strip().lower()
            title = doc.metadata.get('title', 'Untitled')
            preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            
            # Check for exact containment
            if expected_text in doc_text:
                found = True
                best_score = 1.0
            else:
                # Fallback to Levenshtein similarity for fuzzy matching
                score = 1 - (Levenshtein.distance(expected_text, doc_text) / max(len(expected_text), len(doc_text)))
                best_score = max(best_score, score)
                if score > 0.8:  # High similarity threshold
                    found = True
            
            doc_previews.append({
                "title": title,
                "preview": preview,
                "similarity_score": best_score
            })
        
        return {
            "retrieval_match": found,
            "retrieval_score": best_score,
            "num_docs": len(docs),
            "retrieved_docs": doc_previews
        }
        
    except Exception as e:
        print(f"Error in retrieval evaluation: {str(e)}")
        return {
            "retrieval_match": False,
            "retrieval_score": 0.0,
            "num_docs": 0,
            "retrieved_docs": []
        }

@weave.op()
def retrieval_scorer(expected: str, model_output: dict) -> dict:
    """Score the retrieval results."""
    try:
        return {
            "found_doc": model_output.get("retrieval_match", False),
            "retrieval_score": model_output.get("retrieval_score", 0.0),
            "num_docs": model_output.get("num_docs", 0),
            "retrieved_docs": model_output.get("retrieved_docs", [])
        }
    except Exception as e:
        print(f"Error in retrieval scoring: {str(e)}")
        return {
            "found_doc": False,
            "retrieval_score": 0.0,
            "num_docs": 0,
            "retrieved_docs": []
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