# eval_synthetic_qa_weave.py

import os
import json
import asyncio

import weave
from weave import Evaluation

import Levenshtein  # pip install python-Levenshtein

from retrieval_graph import graph  # Your retrieval pipeline
from langchain_core.runnables import RunnableConfig

def load_data(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)

@weave.op()
async def run_retrieval(question: str) -> dict:
    try:
        config = RunnableConfig(
            configurable={
                "user_id": "eval_user",
                "retriever_provider": "pinecone",
            }
        )
        
        # Use ainvoke instead of invoke
        result = await graph.ainvoke(
            {"messages": [("user", question)]},
            config
        )
        
        if not result or "messages" not in result:
            print(f"Warning: No result for question: {question}")
            return {"generated_text": ""}
            
        final_response = result["messages"][-1].content
        return {"generated_text": final_response}
    except Exception as e:
        print(f"Error in retrieval: {str(e)}")
        return {"generated_text": ""}

@weave.op()
def match_score(expected: str, model_output: dict) -> dict:
    try:
        gen_text = model_output.get('generated_text', '').strip().lower() if model_output else ''
        exp_text = expected.strip().lower() if expected else ''
        return {"exact_match": gen_text == exp_text}
    except Exception as e:
        print(f"Error in match_score: {str(e)}")
        return {"exact_match": False}

@weave.op()
def levenshtein_score(expected: str, model_output: dict) -> dict:
    try:
        gen_text = model_output.get('generated_text', '') if model_output else ''
        exp_text = expected if expected else ''
        distance = Levenshtein.distance(exp_text.strip(), gen_text.strip())
        return {"levenshtein_distance": distance}
    except Exception as e:
        print(f"Error in levenshtein_score: {str(e)}")
        return {"levenshtein_distance": -1}

async def main():
    weave.init("mirror-agent-evals")

    data = load_data("data/synthetic_qa_dataset.json")
    subset_data = data[:5]

    examples = []
    for item in subset_data:
        examples.append({
            "question": item["question"],
            "expected": item["answer"]
        })

    evaluation = Evaluation(
        dataset=examples,
        scorers=[match_score, levenshtein_score],
        name="Test Run with First 5 QAs"
    )

    await evaluation.evaluate(
        run_retrieval,
        __weave={"display_name": "Retrieval Pipeline Test Run"}
    )

    print("Evaluation complete. View results in Weave UI or by running weave.watch().")

if __name__ == "__main__":
    asyncio.run(main())