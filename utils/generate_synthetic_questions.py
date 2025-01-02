import os
import json
from dotenv import load_dotenv
from typing import List, Dict
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import random
from tenacity import (
    retry,
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type
)

from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings

# 1) LOAD ENV VARS
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mirror-agent-db")  # default fallback
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")  # Add environment
GPT_4_MODEL_NAME = "gpt-4o-2024-11-20"   # Fixed model name

# Add SYSTEM_PROMPT constant after other constants
SYSTEM_PROMPT = """You are a helpful AI that generates synthetic questions based on the provided document.
These questions should:
1. Be directly answerable from the document content
2. Cover different aspects and difficulty levels
3. Include both factual and conceptual questions
4. Have clear, concise answers
5. Help evaluate a RAG system's retrieval accuracy

Format each Q&A pair to test specific retrieval capabilities."""

# 2) INITIALIZE CLIENTS
pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize OpenAI
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
embeddings = OpenAIEmbeddings()

# Configure requests session with retries
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=[500, 502, 503, 504, 429]
)
session.mount('https://', HTTPAdapter(max_retries=retries))

def fetch_all_documents_from_pinecone(namespace: str = None) -> List[Dict]:
    """
    Returns a list of dicts representing the documents stored in Pinecone using query API.
    Each dict has metadata and possibly the text, if stored as 'text' or 'page_content'.
    """
    try:
        # Get index statistics
        stats = index.describe_index_stats()
        print(f"Index stats: {stats}")
        
        if namespace:
            total_count = stats.namespaces.get(namespace, {}).get("vector_count", 0)
        else:
            total_count = stats.total_vector_count
            
        if total_count == 0:
            print("No documents found in this namespace (or empty index).")
            return []
            
        print(f"\nFetching {total_count} documents...")
        
        # Create a zero vector for querying
        dimension = stats.dimension
        zero_vector = [0.0] * dimension
        
        # Fetch documents in batches
        BATCH_SIZE = 100
        docs = []
        
        for i in range(0, total_count, BATCH_SIZE):
            print(f"Fetching batch {i//BATCH_SIZE + 1}/{(total_count + BATCH_SIZE - 1)//BATCH_SIZE}")
            
            try:
                # Query vectors for the batch
                query_response = index.query(
                    vector=zero_vector,
                    top_k=BATCH_SIZE,
                    include_metadata=True,
                    namespace=namespace
                )
                
                for match in query_response.matches:
                    docs.append({
                        "id": match.id,
                        "metadata": match.metadata,
                        "text": match.metadata.get("text", "") or match.metadata.get("page_content", "")
                    })
                    
            except Exception as batch_error:
                print(f"Error fetching batch {i//BATCH_SIZE + 1}: {batch_error}")
                continue
        
        print(f"Successfully fetched {len(docs)} documents")
        return docs
            
    except Exception as e:
        print(f"Error fetching documents: {e}")
        print(f"Error type: {type(e)}")
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

def test_pinecone_connection():
    """Test the Pinecone connection and print debug information."""
    try:
        print("\nTesting Pinecone connection...")
        print(f"Index name: {PINECONE_INDEX_NAME}")
        print(f"Environment: {PINECONE_ENVIRONMENT}")
        
        # Test describe_index_stats
        stats = index.describe_index_stats()
        print(f"\nIndex stats:")
        print(f"Total vectors: {stats.total_vector_count}")
        print(f"Dimension: {stats.dimension}")
        print(f"Namespaces: {stats.namespaces}")
        
        # Test a simple query
        print("\nTesting simple query...")
        query_response = index.query(
            vector=[0.0] * stats.dimension,
            top_k=1,
            include_metadata=True
        )
        print(f"Query response: {query_response}")
        
        return True
        
    except Exception as e:
        print(f"\nPinecone connection test failed:")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        return False

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception)
)
def generate_questions_with_retry(chat_model, doc_text):
    """Generate questions with retry logic and exponential backoff"""
    try:
        response = chat_model.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Document text: {doc_text}\n\nGenerate questions based on this text.")
        ])
        
        # Add a small random delay between requests
        time.sleep(random.uniform(0.5, 2.0))
        return response
    except Exception as e:
        print(f"Error in generate_questions_with_retry: {str(e)}")
        raise

def process_documents(docs):
    """Process documents with rate limiting"""
    qa_pairs = []
    batch_size = 5  # Process in smaller batches
    
    for i, doc in enumerate(docs, 1):
        print(f"Processing document {i}/{len(docs)}...")
        
        if not doc.get('text'):
            print(f"Skipping doc {doc.get('id')} - no text content found.")
            continue
            
        try:
            response = generate_questions_with_retry(llm, doc['text'])
            # Process response and add to qa_pairs
            # ... existing processing code ...
            
        except Exception as e:
            print(f"Error generating questions for document: {str(e)}")
            continue
            
        # Add batch delay every batch_size documents
        if i % batch_size == 0:
            print(f"Batch complete. Waiting before next batch...")
            time.sleep(random.uniform(2.0, 5.0))
            
    return qa_pairs

def main():
    # Initialize Chat model
    chat_model = ChatOpenAI(
        model="gpt-4",
        temperature=0.7
    )

    # Output file path
    out_file = "synthetic_qa_dataset.json"
    
    # Load existing dataset if it exists
    synthetic_qa_dataset = []
    if os.path.exists(out_file):
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                synthetic_qa_dataset = json.load(f)
            print(f"Loaded {len(synthetic_qa_dataset)} existing Q/A pairs from {out_file}")
        except json.JSONDecodeError:
            print(f"Error loading existing file {out_file}, starting fresh")
    
    # Track processed document IDs
    processed_doc_ids = {qa["doc_id"] for qa in synthetic_qa_dataset}

    # Fetch all docs from Pinecone
    print("Fetching documents from Pinecone...")
    docs = fetch_all_documents_from_pinecone()
    
    if not docs:
        print("No documents found. Please index some documents first.")
        return
    
    print(f"Found {len(docs)} documents. Generating synthetic questions...")
    
    # Process documents in batches
    BATCH_SIZE = 5
    for i, doc in enumerate(docs, 1):
        print(f"Processing document {i}/{len(docs)}...")
        
        # Skip if already processed
        if doc.get("id") in processed_doc_ids:
            print(f"Skipping doc {doc.get('id')} - already processed")
            continue
        
        doc_metadata = doc.get("metadata", {})
        doc_text = doc.get("text") or doc_metadata.get("page_content", "")
        
        if not doc_text:
            print(f"Skipping doc {doc.get('id')} - no text content found")
            continue
        
        try:
            # Generate synthetic QA pairs
            qa_pairs = create_synthetic_questions(doc_metadata, doc_text, chat_model)
            
            # Add document context to pairs
            for pair in qa_pairs:
                pair["doc_id"] = doc.get("id")
                pair["metadata"] = {k:v for k,v in doc_metadata.items() if k != "page_content"}
                synthetic_qa_dataset.append(pair)
            
            # Save after each document
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(synthetic_qa_dataset, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(qa_pairs)} new Q/A pairs for document {doc.get('id')}")
            
            # Add batch delay every BATCH_SIZE documents
            if i % BATCH_SIZE == 0:
                print(f"Batch complete. Waiting before next batch...")
                time.sleep(random.uniform(2.0, 5.0))
                
        except Exception as e:
            print(f"Error processing document {doc.get('id')}: {str(e)}")
            # Continue with next document even if this one fails
            continue
    
    print(f"\nCompleted processing. Total {len(synthetic_qa_dataset)} Q/A pairs across {len(docs)} documents.")
    print(f"Dataset saved to: {out_file}")

if __name__ == "__main__":
    # Test Pinecone connection first
    if not test_pinecone_connection():
        print("Exiting due to Pinecone connection failure")
        exit(1)
        
    # Continue with main script
    print("\nStarting main script execution...")
    main()