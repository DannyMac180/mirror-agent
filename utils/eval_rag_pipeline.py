# eval_synthetic_qa.py

import os
import json
from braintrust import Eval
from autoevals import Levenshtein

from retrieval_graph import graph  # import your main retrieval graph
from langchain_core.runnables import RunnableConfig

from dotenv import load_dotenv

load_dotenv()

BRAINTRUST_API_KEY = os.getenv("BRAINTRUST_API_KEY")

def load_data(json_path: str):
    """Load the synthetic QA dataset JSON."""
    with open(json_path, "r") as f:
        return json.load(f)

async def run_retrieval(question: str) -> str:
    """
    Call your retrieval pipeline (the 'graph' object) with the given question.
    Returns the last AI response string from the pipeline.
    """
    # Configure to use Pinecone
    config = RunnableConfig(
        configurable={
            "user_id": "eval_user",        # Arbitrary ID for this test
            "retriever_provider": "pinecone",
        }
    )

    # Invoke the graph with the question
    result = await graph.ainvoke(
        {"messages": [("user", question)]},
        config
    )

    # Extract the final text from the AI's response
    return result["messages"][-1].content

async def main():
    # Load the synthetic QA dataset
    with open("data/synthetic_qa_dataset.json", "r") as f:
        qa_dataset = json.load(f)

    # Convert dataset to expected format - ONLY FIRST 5 QUESTIONS
    eval_data = [
        {
            "input": qa["question"],
            "expected": qa["answer"]
        }
        for qa in qa_dataset[:5]  # Limit to first 5
    ]

    print(f"Running evaluation on {len(eval_data)} questions...")

    await Eval(
        "mirror-agent",
        task=run_retrieval,
        scores=[Levenshtein],
        data=lambda: eval_data
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())