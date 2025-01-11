#!/usr/bin/env python3
"""
obsidian_indexer.py

Relies solely on `doc.metadata["path"]` for file paths. Keeps LLM logic intact.
"""

import os
import json
import logging
import google.cloud.logging
from google.cloud.logging_v2.handlers import CloudLoggingHandler
from dotenv import load_dotenv
from langchain_community.document_loaders import ObsidianLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import argparse

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

def generate_context(llm, doc_text: str, chunk_text: str) -> str:
    """
    Given the full document text and a chunk of that document, ask Gemini
    to produce a short contextual summary clarifying the chunk's role in the doc.
    """
    prompt = f"""You are given an entire document and one chunk from it:
Document Text:
{doc_text}

Chunk Text:
{chunk_text}

In 1-2 sentences, explain how this chunk relates to the overall document and what role it plays:"""

    response_str = llm.invoke(prompt)
    return response_str.strip()

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
    # 2. Load environment variables and initialize models
    # --------------------------------------------------------------------------
    vault_root = os.getenv("OBSIDIAN_PATH")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    logger.info("Starting Obsidian indexing job...")
    print("Starting Obsidian indexing job...")

    # Keep your Gemini model logic intact
    from langchain_google_genai import GoogleGenerativeAI
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=.7, max_tokens=None, timeout=None, max_retries=2)

    # --------------------------------------------------------------------------
    # 3. Load existing index data
    # --------------------------------------------------------------------------
    indexed_files = load_indexed_files()

    # --------------------------------------------------------------------------
    # 4. Initialize Chroma
    # --------------------------------------------------------------------------
    CHROMA_PERSIST_DIR = os.path.join("data", "chroma")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'mps'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )

    # --------------------------------------------------------------------------
    # 5. Initialize text splitter
    # --------------------------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n---\n", "\n\n", "\n", " ", ""],
        length_function=len,
    )

    # --------------------------------------------------------------------------
    # 6. Load all documents from vault
    # --------------------------------------------------------------------------
    try:
        loader = ObsidianLoader(vault_root)
        all_docs = loader.load()
        logger.info(f"Found {len(all_docs)} documents in vault")
        print(f"Found {len(all_docs)} documents in vault")
    except Exception as e:
        logger.error(f"Failed to load documents from vault: {str(e)}")
        print(f"Failed to load documents from vault: {str(e)}")
        return

    # --------------------------------------------------------------------------
    # 7. Process each document
    # --------------------------------------------------------------------------
    changed_count = 0
    unchanged_count = 0
    error_count = 0

    for doc_index, doc in enumerate(all_docs, 1):
        # Use doc.metadata["path"] as the single source of truth
        file_path = doc.metadata.get("path")
        if not file_path:
            logger.warning(f"No `path` found in doc metadata. Skipping doc: {doc}")
            continue

        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            continue

        record = indexed_files.get(file_path, None)

        # Gather the OS mtime
        try:
            mtime = os.path.getmtime(file_path)
        except (OSError, IOError) as e:
            logger.error(f"Failed to get mtime for {file_path}: {str(e)}")
            continue

        if record and isinstance(record, dict):
            old_mtime = record.get("mtime", 0)
        else:
            old_mtime = 0

        if mtime <= old_mtime:  # unchanged
            unchanged_count += 1
            logger.info(f"Skipping unchanged doc: {file_path}")
            continue

        changed_count += 1
        logger.info(f"Processing changed/new file: {file_path}")

        # Remove old chunks if they exist
        if record and "chunk_ids" in record:
            try:
                vectorstore.delete(ids=record["chunk_ids"])
                logger.info(f"Removed {len(record['chunk_ids'])} old chunks for {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove old chunks for {file_path}: {str(e)}")
                error_count += 1
                continue

        # Split doc into chunks
        try:
            text_chunks = splitter.split_text(doc.page_content)
            chunks_list = []
            for i, chunk_text in enumerate(text_chunks):
                chunk = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"{file_path}-{i}"
                    }
                )
                chunks_list.append(chunk)
            logger.info(f"Split {file_path} into {len(chunks_list)} chunks")
        except Exception as e:
            logger.error(f"Failed to split document {file_path}: {str(e)}")
            print(f"Failed to split document {file_path}: {str(e)}")
            continue

        full_doc_text = doc.page_content

        chunk_ids = []
        processed_chunks = []

        for i, chunk in enumerate(chunks_list):
            try:
                cid = chunk.metadata["chunk_id"]
                chunk_content = chunk.page_content
                context = generate_context(llm, full_doc_text, chunk_content)
                chunk.page_content = chunk_content + "\n\nContext:\n" + context
                chunk_ids.append(cid)
                processed_chunks.append(chunk)
            except Exception as e:
                logger.error(f"Failed to process chunk {i} of {file_path}: {str(e)}")
                continue

        # Add chunks to Chroma
        try:
            vectorstore.add_documents(
                documents=processed_chunks,
                ids=chunk_ids
            )
            logger.info(f"Successfully uploaded {len(processed_chunks)} chunks from {os.path.basename(file_path)} to Chroma")
            print(f"Successfully uploaded {len(processed_chunks)} chunks from {os.path.basename(file_path)} to Chroma")

            indexed_files[file_path] = {
                "mtime": mtime,
                "chunk_ids": chunk_ids,
            }
            save_indexed_files(indexed_files)
        except Exception as e:
            logger.error(f"Failed to upload chunks to Chroma for {file_path}: {str(e)}")
            print(f"Failed to upload chunks to Chroma for {file_path}: {str(e)}")
            continue

    # --------------------------------------------------------------------------
    # 8. Report final stats
    # --------------------------------------------------------------------------
    logger.info(f"Finished processing documents.")
    logger.info(f"Changed/new: {changed_count}")
    logger.info(f"Unchanged: {unchanged_count}")
    logger.info(f"Errors: {error_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Index Obsidian vault documents')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parsed_args = parser.parse_args()
    main(test_mode=parsed_args.test)