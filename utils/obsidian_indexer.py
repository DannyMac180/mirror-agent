#!/usr/bin/env python3
"""
obsidian_indexer.py

Relies solely on `doc.metadata["path"]` for file paths. 
Updated to incorporate:
  1) Doc-level summary (optional)
  2) Bigger chunks & zero overlap
  3) Batch embeddings
  4) Parallel processing

"""
import os
import json
import logging
import google.cloud.logging
from joblib import Parallel, delayed
from google.cloud.logging_v2.handlers import CloudLoggingHandler
from dotenv import load_dotenv
from langchain_community.document_loaders import ObsidianLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import argparse

def generate_doc_summary(llm, doc_text: str) -> str:
    """
    Generate a single doc-level summary in 2-3 lines 
    (much cheaper than chunk-level calls).
    """
    prompt = f"""Summarize the following document in 2-3 lines:\n\n{doc_text}\n"""
    try:
        response_str = llm.invoke(prompt)
        return response_str.strip()
    except Exception as e:
        # fallback or empty summary if there's an error
        return "No summary available."

INDEXED_FILES_JSON = os.path.join("data", "indexed_files.json")

def load_indexed_files() -> dict:
    """
    Load the JSON file that tracks which files have been indexed and their
    last modified times, as well as the chunk IDs in Chroma (so we can remove
    them if changed).
    """
    if not os.path.exists(INDEXED_FILES_JSON):
        return {}
    try:
        with open(INDEXED_FILES_JSON, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading indexed files: {str(e)}")
        return {}

def save_indexed_files(index_data: dict):
    """Persist updated index data to indexed_files.json."""
    try:
        os.makedirs(os.path.dirname(INDEXED_FILES_JSON), exist_ok=True)
        with open(INDEXED_FILES_JSON, 'w') as f:
            json.dump(index_data, f, indent=2)
    except Exception as e:
        print(f"Error saving indexed files: {str(e)}")

def normalize_path(path: str) -> str:
    """
    Normalize a file path to ensure consistent handling across the codebase.
    Converts to absolute path and resolves any symlinks or '..' components.
    """
    return os.path.abspath(os.path.expanduser(path))

def main(test_mode=False):
    # --------------------------------------------------------------------------
    # 1. Load environment variables and setup logging
    # --------------------------------------------------------------------------
    load_dotenv()
    logger = logging.getLogger("obsidian_indexer")
    logger.setLevel(logging.INFO)
    client = google.cloud.logging.Client()
    cloud_handler = CloudLoggingHandler(client, name="obsidian_indexer")
    logger.addHandler(cloud_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # --------------------------------------------------------------------------
    # 2. Load environment variables and initialize LLM + embeddings
    # --------------------------------------------------------------------------
    vault_root = os.getenv("OBSIDIAN_PATH")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Keep your Gemini model logic if needed
    from langchain_google_genai import GoogleGenerativeAI
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=.7, 
        max_tokens=None, 
        timeout=None, 
        max_retries=2
    )
    logger.info("Starting Obsidian indexing job...")
    print("Starting Obsidian indexing job...")

    # --------------------------------------------------------------------------
    # 3. Load existing index data
    # --------------------------------------------------------------------------
    indexed_files = load_indexed_files()

    # --------------------------------------------------------------------------
    # 4. Initialize Chroma
    # --------------------------------------------------------------------------
    CHROMA_PERSIST_DIR = os.path.join("data", "chroma")
    vectorstore = Chroma(
        # We'll pass in embeddings later via add_embeddings()
        # Or if we keep embedding_function here, we can do so. 
        # For batch embedding, we'll manually call embeddings below.
        embedding_function=None,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="obsidian"
    )

    # If your code or environment depends on passing embedding_function to Chroma,
    # revert to embedding_function=embeddings and skip the "batch embedding" approach below.
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'mps'},
        encode_kwargs={'normalize_embeddings': True},
    )

    # --------------------------------------------------------------------------
    # 5. Initialize text splitter
    # --------------------------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0,
        separators=["\n---\n", "\n\n", "\n", " ", ""],
        length_function=len,  # or custom token length function
    )

    # Next, define a function to process a single doc (for parallelization):
    def process_single_document(doc_index, doc, total_docs, indexed_files):
        """
        Process a single document and return tuple of (changed, unchanged, error) counts
        """
        # Use doc.metadata["path"] as the single source of truth
        file_path = doc.metadata.get("path")
        if not file_path:
            logger.warning(f"No `path` found in doc metadata. Skipping doc {doc_index}/{total_docs}: {doc}")
            return (0, 0, 0)  # changed, unchanged, error

        if not file_path or not os.path.exists(file_path):
            logger.error(f"File does not exist ({doc_index}/{total_docs}): {file_path}")
            return (0, 0, 1)

        record = indexed_files.get(file_path, None)

        # Gather the OS mtime
        try:
            mtime = os.path.getmtime(file_path)
        except (OSError, IOError) as e:
            logger.error(f"Failed to get mtime for {file_path} ({doc_index}/{total_docs}): {str(e)}")
            return (0, 0, 1)

        if record and isinstance(record, dict):
            old_mtime = record.get("mtime", 0)
        else:
            old_mtime = 0

        if mtime <= old_mtime:  # unchanged
            logger.info(f"Skipping unchanged doc ({doc_index}/{total_docs}): {file_path}")
            print(f"Skipping unchanged doc ({doc_index}/{total_docs}): {os.path.basename(file_path)}")
            return (0, 1, 0)

        changed_count = 1
        error_count = 0

        logger.info(f"Processing changed/new file ({doc_index}/{total_docs}): {file_path}")
        print(f"Processing ({doc_index}/{total_docs}): {os.path.basename(file_path)}")

        # Remove old chunks if they exist
        if record and "chunk_ids" in record:
            try:
                vectorstore.delete(ids=record["chunk_ids"])
                logger.info(f"Removed {len(record['chunk_ids'])} old chunks for {file_path}")
            except Exception as e:
                logger.error(
                    f"Failed to remove old chunks for {file_path}: {str(e)}"
                )
                return (0, 0, 1)

        # Generate a doc-level summary once
        doc_summary = generate_doc_summary(llm, doc.page_content)

        # Edge case: If doc is empty or only whitespace, let's skip it entirely
        if not doc.page_content.strip():
            logger.warning(f"Document {file_path} is empty. Skipping.")
            return (0, 0, 0)

        # Split doc into chunks
        try:
            text_chunks = splitter.split_text(doc.page_content)
            logger.info(f"Split {file_path} into {len(text_chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to split document {file_path}: {str(e)}")
            print(f"Failed to split document {file_path}: {str(e)}")
            return (0, 0, 1)

        # Build chunk docs
        chunks_list = []
        chunk_ids = []

        for i, chunk_text in enumerate(text_chunks):
            # Skip empty/whitespace-only chunks
            if not chunk_text.strip():
                logger.info(f"Skipping empty chunk #{i} in {file_path}")
                continue

            combined_text = chunk_text + f"\n\nDoc-Level Summary:\n{doc_summary}"
            cid = f"{file_path}-{i}"
            chunk = Document(
                page_content=combined_text,
                metadata={**doc.metadata, "chunk_id": cid}
            )
            chunks_list.append(chunk)
            chunk_ids.append(cid)

        # If all chunks were empty, skip indexing entirely.
        if not chunks_list:
            logger.warning(f"All chunks empty for {file_path}. Skipping indexing.")
            return (0, 0, 0)

        # BATCH EMBEDDINGS:
        # Manually embed all chunk texts at once
        try:
            all_texts = [c.page_content for c in chunks_list]
            embeddings_list = embeddings.embed_documents(all_texts)  # returns List[List[float]]
        except Exception as e:
            logger.error(
                f"Failed to embed chunks for {file_path} with error: {str(e)}"
            )
            return (0, 0, 1)

        # If embeddings_list is empty or contains no valid embeddings, skip
        if not embeddings_list:
            logger.warning(f"No embeddings returned for {file_path}. Skipping indexing.")
            return (0, 0, 0)

        # Instead of vectorstore.add_embeddings(), directly call the underlying ChromaDB collection
        try:
            # Access the underlying chromadb.Collection
            collection = vectorstore._collection
            # Use the low-level .add() method
            collection.add(
                embeddings=embeddings_list,
                documents=all_texts,
                metadatas=[c.metadata for c in chunks_list],
                ids=chunk_ids
            )
            logger.info(
                f"Successfully uploaded {len(chunks_list)} chunks from {os.path.basename(file_path)} to Chroma"
            )
            print(f"Successfully uploaded {len(chunks_list)} chunks from {os.path.basename(file_path)} to Chroma")
        except Exception as e:
            logger.error(f"Failed to upload chunks to Chroma for {file_path}: {str(e)}")
            print(f"Failed to upload chunks to Chroma for {file_path}: {str(e)}")
            return (0, 0, 1)

        # Update indexed_files
        indexed_files[file_path] = {
            "mtime": mtime,
            "chunk_ids": chunk_ids,
        }
        save_indexed_files(indexed_files)  # persist updated index

        return (1, 0, 0)  # changed, unchanged, error

    # Load the doc set
    try:
        loader = ObsidianLoader(vault_root)
        all_docs = loader.load()
        total_docs = len(all_docs)
        logger.info(f"Found {total_docs} documents in vault")
        print(f"Found {total_docs} documents in vault")
    except Exception as e:
        logger.error(f"Failed to load documents from vault: {str(e)}")
        print(f"Failed to load documents from vault: {str(e)}")
        return

    # Instead of sequential, use joblib for parallel processing:
    # (Adjust n_jobs based on your CPU cores or environment)
    num_jobs = min(4, os.cpu_count() or 1)
    results = Parallel(n_jobs=num_jobs, backend="threading")(
        delayed(process_single_document)(doc_index, doc, total_docs, indexed_files)
        for doc_index, doc in enumerate(all_docs, 1)
    )

    # Aggregate results
    changed_count = sum(r[0] for r in results)
    unchanged_count = sum(r[1] for r in results)
    error_count = sum(r[2] for r in results)

    # --------------------------------------------------------------------------
    # 8. Report final stats
    # --------------------------------------------------------------------------
    logger.info(f"Finished processing documents.")
    logger.info(f"Changed/new: {changed_count}")
    logger.info(f"Unchanged: {unchanged_count}")
    logger.info(f"Errors: {error_count}")

    print(f"Indexing job complete.\nChanged/new: {changed_count}\nUnchanged: {unchanged_count}\nErrors: {error_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Index Obsidian vault documents')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()
    main(test_mode=args.test)