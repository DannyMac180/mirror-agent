import os
import json
import time
from pathlib import Path
from typing import Dict, List

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import ObsidianLoader
from langchain.docstore.document import Document

# Load .env variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "gcp-starter")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mirror-agent")

# Path to your Obsidian vault
OBSIDIAN_PATH = '/Users/danielmcateer/Library/Mobile Documents/iCloud~md~obsidian/Documents/Ideaverse'

# Local JSON for tracking which files have been indexed 
DB_RECORDS_PATH = Path(__file__).parent.parent / "indexed_files.json"

def load_indexed_records() -> Dict[str, float]:
    """
    Loads the dictionary { file_path: last_modified_timestamp } 
    from a JSON file. If the file doesn't exist or is invalid, returns an empty dict.
    """
    if not DB_RECORDS_PATH.exists():
        print("No existing index records found. This appears to be the first run.")
        return {}
    try:
        with open(DB_RECORDS_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
            print(f"Loaded {len(records)} existing file records.")
            return records
    except Exception as e:
        print(f"Error loading index records: {e}")
        return {}

def save_indexed_records(records: Dict[str, float]) -> None:
    """
    Saves the dictionary of file->timestamp so we can track changes for the next run.
    """
    try:
        with open(DB_RECORDS_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"Saved {len(records)} file records to {DB_RECORDS_PATH}")
    except Exception as e:
        print(f"Error saving index records: {e}")

def upsert_obsidian_vault(vault_path: str) -> None:
    """
    1. Loads all markdown from an Obsidian vault
    2. Upserts only the changed/new files to Pinecone
    3. Tracks files with 'indexed_files.json' so we only upsert deltas next time
    """

    # Initialize Pinecone
    if not PINECONE_API_KEY:
        raise ValueError("Missing Pinecone API key. Please set PINECONE_API_KEY in your .env file.")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    try:
        # Try to get the index first
        index = pc.Index(INDEX_NAME)
        print(f"Connected to existing index: {INDEX_NAME}")
    except Exception:
        # If index doesn't exist, create it
        print(f"Creating new index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME, 
            dimension=1536,   # for text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(
                cloud="gcp",  # Using GCP for starter environment
                region="us-central1"  # GCP region format
            )
        )
        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        index = pc.Index(INDEX_NAME)

    # Create the embeddings + VectorStore
    embeddings = OpenAIEmbeddings()  # requires OPENAI_API_KEY in your .env
    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    # Load existing upsert records
    upsert_records = load_indexed_records()
    is_first_run = not DB_RECORDS_PATH.exists()

    # Load documents from Obsidian
    input_path = Path(vault_path).expanduser().resolve()
    if not input_path.exists():
        raise ValueError(f"Vault path does not exist: {vault_path}")

    print(f"Loading markdown files from: {input_path}")
    loader = ObsidianLoader(str(input_path), collect_metadata=True)
    docs: List[Document] = loader.load()
    print(f"Loaded {len(docs)} documents from vault.")

    docs_to_upsert = []
    updated_records = dict(upsert_records)  # copy so we don't mutate in-place

    # Check each doc for changes
    for doc in docs:
        file_path = doc.metadata.get("source", "")
        if not file_path:
            continue
        
        path_obj = Path(file_path)
        if not path_obj.exists():
            continue  # skip if the file is somehow missing

        mtime = path_obj.stat().st_mtime
        prev_mtime = upsert_records.get(file_path, 0.0)
        
        # If it's the first run or the file has been modified, add it to upsert list
        if is_first_run or mtime > prev_mtime:
            docs_to_upsert.append(doc)
            updated_records[file_path] = mtime

    # Upsert only if we have deltas
    if not docs_to_upsert:
        print("No new or changed markdown files to upsert.")
        return

    print(f"Upserting {len(docs_to_upsert)} documents to Pinecone...")
    # If you want chunking, do it here with a TextSplitter before add_documents()
    vectorstore.add_documents(docs_to_upsert)

    # Update local DB
    save_indexed_records(updated_records)
    print("Upsert complete. Updated local records.")

# If you want to run this manually: 
if __name__ == "__main__":
    upsert_obsidian_vault(OBSIDIAN_PATH)
    print("Done indexing. You can now query Pinecone from your retrieval logic!")