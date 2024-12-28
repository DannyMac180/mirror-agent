import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mirror-agent")

def clear_pinecone_index():
    """Delete all records from the Pinecone index."""
    
    if not PINECONE_API_KEY:
        raise ValueError("Missing Pinecone API key. Please set PINECONE_API_KEY in your .env file.")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    try:
        # Connect to the index
        index = pc.Index(INDEX_NAME)
        print(f"Connected to index: {INDEX_NAME}")
        
        # Delete all vectors
        index.delete(delete_all=True)
        print("Successfully deleted all records from the index.")
        
    except Exception as e:
        print(f"Error clearing index: {e}")

if __name__ == "__main__":
    clear_pinecone_index() 