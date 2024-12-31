import os
import json
from dotenv import load_dotenv
from typing import List, Dict

from pinecone import Pinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# 1) LOAD ENV VARS
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mirror-agent")  # default fallback
GPT_4_MODEL_NAME = "gpt-4"   # Using standard GPT-4 model

# 2) INITIALIZE CLIENTS
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize OpenAI
llm = ChatOpenAI(temperature=0.7)
embeddings = OpenAIEmbeddings()

def fetch_all_documents_from_pinecone(namespace: str = None) -> List[Dict]:
    """
    Returns a list of dicts representing the documents stored in Pinecone. 
    Each dict has metadata and possibly the text, if stored as 'text' or 'page_content'.
    """
    try:
        # Get index statistics
        stats = index.describe_index_stats()
        
        if namespace:
            total_count = stats.namespaces.get(namespace, {}).get("vector_count", 0)
        else:
            total_count = stats.total_vector_count
            
        if total_count == 0:
            print("No documents found in this namespace (or empty index).")
            return []
            
        # Load the indexed files record to get IDs
        if os.path.exists("indexed_files.json"):
            with open("indexed_files.json", "r") as f:
                indexed_files = json.load(f)
            
            # Extract IDs from indexed files
            doc_ids = [doc["id"] for doc in indexed_files]
            
            # Fetch documents in batches
            BATCH_SIZE = 50
            docs = []
            
            for i in range(0, len(doc_ids), BATCH_SIZE):
                batch_ids = doc_ids[i:i + BATCH_SIZE]
                # Fetch vectors for the batch
                fetch_response = index.fetch(ids=batch_ids, namespace=namespace)
                
                for vector_id, vector in fetch_response.vectors.items():
                    docs.append({
                        "id": vector_id,
                        "metadata": vector.metadata,
                        "text": vector.metadata.get("page_content", "")  # Assuming content is stored in metadata
                    })
            
            return docs
        else:
            print("No indexed_files.json found. Please index documents first.")
            return []
            
    except Exception as e:
        print(f"Error fetching documents: {e}")
        return []

def create_synthetic_questions(doc_metadata: Dict, doc_text: str, chat_model: ChatOpenAI) -> List[Dict]:
    """
    Given the doc metadata and doc_text from Pinecone, 
    generate synthetic Q/A pairs using GPT-4 via LangChain.
    
    Creates 3-5 synthetic questions per doc.
    """
    system_content = """You are a helpful AI that generates synthetic questions based on the provided document.
These questions should:
1. Be directly answerable from the document content
2. Cover different aspects and difficulty levels
3. Include both factual and conceptual questions
4. Have clear, concise answers
5. Help evaluate a RAG system's retrieval accuracy

Format each Q&A pair to test specific retrieval capabilities."""
    
    metadata_string = ", ".join([f"{k}: {v}" for k, v in doc_metadata.items() if k != "page_content"])
    
    user_content = f"""
Document Metadata:
{metadata_string}

Document Text:
{doc_text[:2000]}  # First 2000 chars for context

Generate 3-5 synthetic questions and answers that can be used to test retrieval accuracy.
Return in JSON format:
[
  {{
    "question": "...",
    "answer": "...",
    "type": "factual|conceptual|analytical"
  }},
  ...
]
"""
    
    try:
        response = chat_model([
            SystemMessage(content=system_content),
            HumanMessage(content=user_content)
        ])
        
        qa_list = json.loads(response.content)
        if isinstance(qa_list, list):
            return qa_list
        return []
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error generating questions for document: {e}")
        return []

def main():
    # Initialize Chat model
    chat_model = ChatOpenAI(
        model=GPT_4_MODEL_NAME,
        temperature=0.7
    )

    # Fetch all docs from Pinecone
    print("Fetching documents from Pinecone...")
    docs = fetch_all_documents_from_pinecone()
    
    if not docs:
        print("No documents found. Please index some documents first.")
        return
    
    print(f"Found {len(docs)} documents. Generating synthetic questions...")
    synthetic_qa_dataset = []
    
    for i, doc in enumerate(docs, 1):
        print(f"Processing document {i}/{len(docs)}...")
        
        doc_metadata = doc.get("metadata", {})
        doc_text = doc.get("text") or doc_metadata.get("page_content", "")
        
        if not doc_text:
            print(f"Skipping doc {doc.get('id')} - no text content found.")
            continue
        
        # Generate synthetic QA pairs
        qa_pairs = create_synthetic_questions(doc_metadata, doc_text, chat_model)
        
        # Add to dataset with document context
        for pair in qa_pairs:
            pair["doc_id"] = doc.get("id")
            pair["metadata"] = {k:v for k,v in doc_metadata.items() if k != "page_content"}
            synthetic_qa_dataset.append(pair)
    
    # Save the dataset
    out_file = "synthetic_qa_dataset.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(synthetic_qa_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\nGenerated {len(synthetic_qa_dataset)} synthetic Q/A pairs across {len(docs)} documents.")
    print(f"Dataset saved to: {out_file}")

if __name__ == "__main__":
    main()