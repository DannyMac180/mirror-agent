#!/usr/bin/env python3
"""
obsidian_indexer.py

 - Loads environment variables from .env
 - Checks data/indexed_files.json for known docs + last modified times
 - Loads/chunks Obsidian .md files using LangChain's ObsidianLoader
 - Adds contextual retrieval logic by having Google's Gemini model summarize each chunk in context
 - Indexes changed or new docs into a persistent Chroma DB (collection="obsidian")
 - Logs to both stdout and Google Cloud Logging
"""

import os
import json
import time
import logging
from dotenv import load_dotenv
from datetime import datetime

from langchain_community.document_loaders import ObsidianLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI

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


def normalize_path(path: str) -> str:
    """
    Normalize a file path to ensure consistent handling across the codebase.
    Converts to absolute path and resolves any symlinks or '..' components.
    """
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def generate_context(llm, doc_text: str, chunk_text: str) -> str:
    """
    Given the full document text and a chunk of that document,
    ask Gemini to produce a short contextual summary
    that clarifies the chunk's role/meaning within the doc.
    """
    prompt = f"""You are given an entire document and one chunk from it:
Document Text:
{doc_text}

Chunk:
{chunk_text}

Please provide a short, succinct context or summary of this chunk's role in the document.
Only return the short context text, nothing else.
"""
    response = llm.invoke(prompt)
    return response.content.strip()


def main():
    # --------------------------------------------------------------------------
    # 1. Load environment variables
    # --------------------------------------------------------------------------
    load_dotenv()
    OBSIDIAN_PATH = normalize_path(os.getenv("OBSIDIAN_PATH", "/Users/danielmcateer/Library/Mobile Documents/iCloud~md~obsidian/Documents/Ideaverse"))
    CHROMA_DB_DIR = normalize_path(os.getenv("CHROMA_DB_DIR", "./chroma_db"))
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # --------------------------------------------------------------------------
    # 2. Setup GCP Logging + Standard Logging
    # --------------------------------------------------------------------------
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("obsidian_indexer")
    
    logger.info("Starting Obsidian indexing job...")
    print("Starting Obsidian indexing job...")

    # Initialize Gemini model
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # --------------------------------------------------------------------------
    # 3. Load existing index data
    # --------------------------------------------------------------------------
    indexed_files = load_indexed_files()
    logger.info(f"Loaded index data for {len(indexed_files)} previously indexed files.")
    print(f"Loaded index data for {len(indexed_files)} previously indexed files.")

    # --------------------------------------------------------------------------
    # 4. Load Obsidian docs and initialize Chroma
    # --------------------------------------------------------------------------
    logger.info(f"Loading documents from vault: {OBSIDIAN_PATH}")
    print(f"Loading documents from vault: {OBSIDIAN_PATH}")

    # Initialize embeddings and Chroma first
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    vectorstore = Chroma(
        collection_name="obsidian",
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

    # Then load and process documents
    loader = ObsidianLoader(OBSIDIAN_PATH)
    all_docs = loader.load()  # returns a list of Documents

    # Update document metadata to use absolute paths
    for doc in all_docs:
        relative_path = doc.metadata.get("source") or doc.metadata.get("file_path")
        if relative_path:
            try:
                # Convert relative path to absolute path and normalize
                absolute_path = normalize_path(os.path.join(OBSIDIAN_PATH, relative_path))
                # Verify the file exists and is within the Obsidian vault
                if not os.path.exists(absolute_path):
                    logger.error(f"File does not exist: {relative_path}")
                    continue
                if not absolute_path.startswith(OBSIDIAN_PATH):
                    logger.error(f"File path {relative_path} resolves outside of Obsidian vault")
                    continue
                doc.metadata["source"] = absolute_path
                doc.metadata["file_path"] = absolute_path
                doc.metadata["relative_path"] = os.path.relpath(absolute_path, OBSIDIAN_PATH)
            except Exception as e:
                logger.error(f"Error processing file path {relative_path}: {str(e)}")
                continue

    # Create a set of current file paths for checking deleted files
    current_files = {doc.metadata.get("source") 
                    for doc in all_docs 
                    if doc.metadata.get("source") and os.path.exists(doc.metadata["source"])}
    
    # Check for deleted files
    deleted_files = set(indexed_files.keys()) - current_files
    if deleted_files:
        logger.info(f"Found {len(deleted_files)} files that have been deleted from Obsidian")
        print(f"Found {len(deleted_files)} files that have been deleted from Obsidian")
        
        for file_path in deleted_files:
            try:
                # Remove chunks from Chroma
                chunk_ids = indexed_files[file_path].get("chunk_ids", [])
                if chunk_ids:
                    vectorstore.delete(ids=chunk_ids)
                    logger.info(f"Deleted {len(chunk_ids)} chunks for removed file: {file_path}")
                    print(f"Deleted {len(chunk_ids)} chunks for removed file: {file_path}")
                # Remove from indexed files
                del indexed_files[file_path]
                save_indexed_files(indexed_files)
            except Exception as e:
                logger.error(f"Error cleaning up deleted file {file_path}: {str(e)}")
                print(f"Error cleaning up deleted file {file_path}: {str(e)}")

    logger.info(f"Found {len(all_docs)} total documents in Obsidian vault")
    print(f"Found {len(all_docs)} total documents in Obsidian vault")
    
    # --------------------------------------------------------------------------
    # 5. Initialize text splitter
    # --------------------------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    # --------------------------------------------------------------------------
    # 6. Iterate through docs, skip unchanged, reindex changed
    # --------------------------------------------------------------------------
    changed_count = 0
    unchanged_count = 0
    error_count = 0

    total_docs = len(all_docs)
    for doc_index, doc in enumerate(all_docs, 1):
        file_path = doc.metadata.get("source") or doc.metadata.get("file_path")
        if not file_path:
            error_count += 1
            logger.error(f"Document {doc_index} has no file path, skipping")
            continue

        # Get relative path for display
        try:
            relative_path = os.path.relpath(file_path, OBSIDIAN_PATH)
        except ValueError as e:
            error_count += 1
            logger.error(f"Invalid path relationship between {file_path} and {OBSIDIAN_PATH}: {str(e)}")
            continue

        print(f"\nProcessing document {doc_index}/{total_docs}: {relative_path}")

        try:
            stat = os.stat(file_path)
            mtime = stat.st_mtime
            # Round to nearest second for more reliable comparison
            mtime = int(mtime)
        except (FileNotFoundError, PermissionError) as e:
            error_count += 1
            logger.error(f"Error accessing file {relative_path}: {str(e)}")
            continue

        record = indexed_files.get(file_path)
        
        if record is not None:
            old_mtime = int(record.get("last_modified", 0))
            if mtime <= old_mtime:
                unchanged_count += 1
                logger.info(f"Skipping unchanged file: {relative_path} (last modified: {datetime.fromtimestamp(mtime)})")
                print(f"  → Skipping unchanged file (last modified: {datetime.fromtimestamp(mtime)})")
                continue
            logger.info(f"Processing changed file: {relative_path} (modified: {datetime.fromtimestamp(mtime)})")
            print(f"  → File changed (last modified: {datetime.fromtimestamp(mtime)})")
        else:
            logger.info(f"Processing new file: {relative_path} (created: {datetime.fromtimestamp(mtime)})")
            print(f"  → New file detected (created: {datetime.fromtimestamp(mtime)})")
            old_mtime = 0

        # Doc changed or new
        changed_count += 1
        logger.info(f"Processing changed/new file: {file_path}")

        # If we have chunk IDs from before, remove them from Chroma
        old_chunk_ids = record.get("chunk_ids", []) if record else []
        if old_chunk_ids:
            try:
                # Remove outdated embeddings if file changed
                vectorstore.delete(ids=old_chunk_ids)
                logger.info(f"Deleted {len(old_chunk_ids)} old chunks for {file_path}")
                print(f"Deleted {len(old_chunk_ids)} old chunks for {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete old chunks for {file_path}: {str(e)}")
                print(f"Failed to delete old chunks for {file_path}: {str(e)}")
                continue

        # Split doc into chunks
        try:
            chunks = splitter.split_documents([doc])
            logger.info(f"Split {file_path} into {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to split document {file_path}: {str(e)}")
            print(f"Failed to split document {file_path}: {str(e)}")
            continue

        # We'll gather the doc's full text once
        full_doc_text = doc.page_content

        chunk_ids = []
        processed_chunks = []
        
        # Process each chunk with Gemini context
        for i, chunk in enumerate(chunks):
            try:
                # Build a stable ID
                cid = f"{file_path}-{i}"
                # Generate short context text from Gemini
                context = generate_context(llm, full_doc_text, chunk.page_content)
                # Append the context to the chunk so it gets embedded
                chunk.page_content = chunk.page_content + "\n\nContext:\n" + context

                chunk.metadata["id"] = cid
                chunk_ids.append(cid)
                processed_chunks.append(chunk)
            except Exception as e:
                logger.error(f"Failed to process chunk {i} of {file_path}: {str(e)}")
                print(f"Failed to process chunk {i} of {file_path}: {str(e)}")
                continue

        if not processed_chunks:
            logger.error(f"No chunks were successfully processed for {file_path}")
            print(f"No chunks were successfully processed for {file_path}")
            continue

        # Add chunks to Chroma
        try:
            vectorstore.add_documents(
                documents=processed_chunks,
                ids=chunk_ids
            )
            logger.info(f"Successfully uploaded {len(processed_chunks)} chunks from {os.path.basename(file_path)} to Chroma")
            print(f"Successfully uploaded {len(processed_chunks)} chunks from {os.path.basename(file_path)} to Chroma")
            
            # Update and save the record immediately after successful upload
            indexed_files[file_path] = {
                "last_modified": mtime,
                "chunk_ids": chunk_ids,
            }
            save_indexed_files(indexed_files)
            logger.info(f"Updated index record for {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"Failed to upload chunks to Chroma for {file_path}: {str(e)}")
            print(f"Failed to upload chunks to Chroma for {file_path}: {str(e)}")
            continue

    # --------------------------------------------------------------------------
    # 7. Persist changes to Chroma
    # --------------------------------------------------------------------------
    vectorstore.persist()

    # --------------------------------------------------------------------------
    # 8. Logging summary
    # --------------------------------------------------------------------------
    # Get total count of documents in Chroma
    total_docs_in_chroma = len(vectorstore.get()['ids'])
    
    summary_msg = (
        f"Indexing job complete. "
        f"New/changed files: {changed_count}, "
        f"Unchanged/skipped: {unchanged_count}, "
        f"Errors: {error_count}. "
        f"Total documents in Chroma: {total_docs_in_chroma}. "
        "Chroma persisted successfully."
    )
    logger.info(summary_msg)
    print(summary_msg)


if __name__ == "__main__":
    main()