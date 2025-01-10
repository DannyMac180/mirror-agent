import chromadb
import os
from dotenv import load_dotenv

def test_chroma_retrieval():
    # Load environment variables
    load_dotenv()
    CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # Get the collection
    collection = client.get_collection("obsidian")

    # Test query
    query = "Vervaeke"
    results = collection.query(
        query_texts=[query],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    print(f"\nQuery: {query}\n")
    print("Results:")
    print("-" * 80)

    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        print(f"\nResult {i} (Distance Score: {distance:.4f}):")
        print("-" * 40)
        print("Content:")
        print(doc)
        print("\nMetadata:")
        print(metadata)
        print("-" * 80)

if __name__ == "__main__":
    test_chroma_retrieval()
