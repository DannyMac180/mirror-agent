#!/usr/bin/env python3
"""
obsidian_indexer.py

- Loads environment variables from .env
- Checks data/indexed_files.json for known docs + last modified times
- Loads/chunks Obsidian .md files using LangChain's ObsidianLoader
- Indexes changed or new docs into a persistent Chroma DB (collection="obsidian")
- Logs to both stdout and Google Cloud Logging
"""

import os
import json
import time
import logging
from dotenv import load_dotenv

from langchain_community.document_loaders import ObsidianLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ------------------------------------------------------------------------------
# Constants / Globals
# ------------------------------------------------------------------------------
INDEXED_FILES_JSON = os.path.join("data", "indexed_files.json")


def load_indexed_files() -> dict:
    """
    Load the JSON file that tracks which files have been indexed and their
    last modified times, as well as the chunk IDs in Chroma (so we can remove
    them if changed).
    """
    if not os.path.exists(INDEXED_FILES_JSON):
        return {}
    with open(INDEXED_FILES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_indexed_files(index_data: dict):
    """Persist updated index data to indexed_files.json."""
    os.makedirs(os.path.dirname(INDEXED_FILES_JSON), exist_ok=True)
    with open(INDEXED_FILES_JSON, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)


def main():
    # --------------------------------------------------------------------------
    # 1. Load environment variables
    # --------------------------------------------------------------------------
    load_dotenv()
    OBSIDIAN_PATH = os.getenv("OBSIDIAN_PATH", "/Users/danielmcateer/Library/Mobile Documents/iCloud~md~obsidian/Documents/Ideaverse")
    CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # --------------------------------------------------------------------------
    # 2. Setup GCP Logging + Standard Logging
    # --------------------------------------------------------------------------
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("obsidian_indexer")
    
    logger.info("Starting Obsidian indexing job...")
    print("Starting Obsidian indexing job...")

    # --------------------------------------------------------------------------
    # 3. Load existing index data
    # --------------------------------------------------------------------------
    indexed_files = load_indexed_files()
    logger.info(f"Loaded index data for {len(indexed_files)} files.")
    print(f"Loaded index data for {len(indexed_files)} files from JSON.")

    # --------------------------------------------------------------------------
    # 4. Load Obsidian docs
    # --------------------------------------------------------------------------
    logger.info(f"Loading documents from vault: {OBSIDIAN_PATH}")
    print(f"Loading documents from vault: {OBSIDIAN_PATH}")

    loader = ObsidianLoader(OBSIDIAN_PATH)
    all_docs = loader.load()  # returns a list of Documents
    
    # Limit to first 5 documents for testing
    test_docs = all_docs[:5]
    logger.info(f"Loaded {len(all_docs)} total documents, testing with first {len(test_docs)}")
    print(f"Loaded {len(all_docs)} total documents, testing with first {len(test_docs)}")
    
    # Print test document paths
    for i, doc in enumerate(test_docs):
        file_path = doc.metadata.get("source") or doc.metadata.get("file_path")
        print(f"Test document {i+1}: {file_path}")

    # --------------------------------------------------------------------------
    # 5. Initialize text splitter, embeddings, and Chroma
    # --------------------------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    vectorstore = Chroma(
        collection_name="obsidian",
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

    # --------------------------------------------------------------------------
    # 6. Iterate through docs, skip unchanged, reindex changed
    # --------------------------------------------------------------------------
    changed_count = 0
    unchanged_count = 0

    for doc in test_docs:  
        # Typically doc.metadata["source"] or doc.metadata["file_path"] is the path
        file_path = doc.metadata.get("source") or doc.metadata.get("file_path")
        if not file_path:
            # If we can't locate the path, skip
            continue

        # Get last modified time from the filesystem
        # If doc.metadata doesn't have it, we can do an os.stat call:
        try:
            stat = os.stat(file_path)
            mtime = stat.st_mtime
        except FileNotFoundError:
            # Possibly the doc is from some other location or ephemeral
            # We'll just index it
            mtime = time.time()

        record = indexed_files.get(file_path)
        if record is not None:
            old_mtime = record.get("last_modified", 0)
        else:
            old_mtime = 0

        # Check if doc changed
        if mtime <= old_mtime:
            # This doc is unchanged; skip re-indexing
            unchanged_count += 1
            continue

        # Doc changed or new
        changed_count += 1
        logger.info(f"Re-indexing changed file: {file_path}")
        print(f"Re-indexing changed file: {file_path}")

        # If we have chunk IDs from before, remove them from Chroma
        old_chunk_ids = record.get("chunk_ids", []) if record else []
        if old_chunk_ids:
            vectorstore.delete(ids=old_chunk_ids)
            logger.info(f"Deleted {len(old_chunk_ids)} old chunks for {file_path}")
            print(f"Deleted {len(old_chunk_ids)} old chunks for {file_path}")

        # Split doc into chunks
        chunks = splitter.split_documents([doc])
        # We can create our own chunk IDs if we like, e.g. file_path + index
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            # Build a stable ID
            cid = f"{file_path}-{i}"
            chunk.metadata["id"] = cid
            chunk_ids.append(cid)

        # Add chunks to Chroma (pass ids=...)
        vectorstore.add_documents(
            documents=chunks,
            ids=chunk_ids
        )

        # Update the record in memory
        indexed_files[file_path] = {
            "last_modified": mtime,
            "chunk_ids": chunk_ids,
        }

    # --------------------------------------------------------------------------
    # 7. Persist changes to Chroma & Save updated index data
    # --------------------------------------------------------------------------
    vectorstore.persist()
    save_indexed_files(indexed_files)

    # --------------------------------------------------------------------------
    # 8. Logging summary
    # --------------------------------------------------------------------------
    summary_msg = (
        f"Indexing job complete. "
        f"New/changed files: {changed_count}, "
        f"Unchanged/skipped: {unchanged_count}. "
        "Chroma persisted successfully."
    )
    logger.info(summary_msg)
    print(summary_msg)


if __name__ == "__main__":
    main()