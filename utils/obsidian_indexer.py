import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import re
import psutil

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from utils.gcp_logging import get_logger

# Load .env variables
load_dotenv()

# Initialize GCP logger
logger = get_logger("mirror-agent")

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def chunk_markdown(content: str, source_path: str) -> List[Document]:
    """Split markdown content into chunks while preserving header context.
    
    Args:
        content: Raw markdown content
        source_path: Path to source file for metadata
        
    Returns:
        List of Document objects with chunks
    """
    try:
        # First split by headers
        headers_to_split_on = [
            ("#", "header1"),
            ("##", "header2"),
            ("###", "header3"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
            return_each_line=False
        )
        header_splits = markdown_splitter.split_text(content)
        
        # Then split large sections further by size
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        
        final_chunks = []
        for doc in header_splits:
            # Get header context
            header_context = {
                k: v for k, v in doc.metadata.items() 
                if k.startswith('header')
            }
            
            # Split large sections further
            if len(doc.page_content) > CHUNK_SIZE:
                smaller_chunks = text_splitter.split_text(doc.page_content)
                for i, chunk in enumerate(smaller_chunks):
                    final_chunks.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": source_path,
                                "chunk_index": i,
                                **header_context
                            }
                        )
                    )
            else:
                doc.metadata["source"] = source_path
                final_chunks.append(doc)
                
        if logger:
            logger.log_info(f"Split document into {len(final_chunks)} chunks", {
                "source": source_path,
                "num_chunks": len(final_chunks)
            })
            
        return final_chunks
        
    except Exception as e:
        if logger:
            logger.log_error(e, {
                "context": "chunking_markdown",
                "source": source_path
            })
        raise

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "gcp-starter")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mirror-agent")

# Path to your Obsidian vault
OBSIDIAN_PATH = '/Users/danielmcateer/Library/Mobile Documents/iCloud~md~obsidian/Documents/Ideaverse'

# Local JSON for tracking which files have been indexed 
DB_RECORDS_PATH = Path(__file__).parent.parent / "data" / "indexed_files.json"

def count_markdown_files(vault_path: str) -> int:
    """Count all markdown files in the Obsidian vault.
    
    Args:
        vault_path (str): Path to the Obsidian vault
        
    Returns:
        int: Number of markdown files found
    """
    path = Path(vault_path).expanduser().resolve()
    if not path.exists():
        if logger:
            logger.log_error(ValueError(f"Vault path does not exist: {vault_path}"))
        raise ValueError(f"Vault path does not exist: {vault_path}")
        
    count = 0
    for root, _, files in os.walk(path):
        count += sum(1 for f in files if f.endswith('.md'))
    return count

# Print and log the count when module is loaded
total_files = count_markdown_files(OBSIDIAN_PATH)
print(f"\nFound {total_files} markdown files in Obsidian vault at: {OBSIDIAN_PATH}\n")
if logger:
    logger.log_start_indexing(OBSIDIAN_PATH, total_files)

def load_indexed_records() -> Dict[str, float]:
    """
    Loads the dictionary { file_path: last_modified_timestamp } 
    from a JSON file. If the file doesn't exist or is invalid, returns an empty dict.
    """
    if not DB_RECORDS_PATH.exists():
        print("No existing index records found. This appears to be the first run.")
        if logger:
            logger.log_warning("First run detected - no existing index records", {
                "records_path": str(DB_RECORDS_PATH)
            })
        return {}
    try:
        with open(DB_RECORDS_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
            print(f"Loaded {len(records)} existing file records.")
            return records
    except Exception as e:
        print(f"Error loading index records: {e}")
        if logger:
            logger.log_error(e, {
                "context": "loading_index_records",
                "records_path": str(DB_RECORDS_PATH)
            })
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
        if logger:
            logger.log_error(e, {
                "context": "saving_index_records",
                "records_path": str(DB_RECORDS_PATH)
            })

def get_performance_metrics() -> Dict[str, Any]:
    """Get current performance metrics."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "memory_usage_mb": memory_info.rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "thread_count": process.num_threads()
    }

def enrich_obsidian_metadata(doc: Document) -> Document:
    """Add Obsidian-specific metadata to the document."""
    try:
        # Get the original metadata
        metadata = doc.metadata.copy()
        
        # Extract title from filename if not present
        if "title" not in metadata and "source" in metadata:
            metadata["title"] = Path(metadata["source"]).stem
            
        # Truncate content for metadata
        content = doc.page_content[:500]  # First 500 chars for preview
        
        # Extract links and tags from content
        links = re.findall(r'\[\[(.*?)\]\]', content)
        tags = re.findall(r'#(\w+)', content)
        
        # Keep only essential metadata within size limits
        filtered_metadata = {
            "title": metadata.get("title", "")[:50],  # Limit title length
            "path": str(Path(metadata.get("relative_path", "")).stem)[:100],  # Just filename
            "links": [l[:50] for l in links[:5]],  # Keep only first 5 links, truncated
            "tags": [t[:20] for t in tags[:5]],  # Keep only first 5 tags, truncated
            "preview": content[:200]  # Shorter preview
        }
        
        # Calculate total metadata size
        metadata_str = str(filtered_metadata)
        if len(metadata_str.encode('utf-8')) > 40000:  # Leave some buffer
            # If still too large, further reduce
            filtered_metadata["preview"] = filtered_metadata["preview"][:100]
            filtered_metadata["links"] = filtered_metadata["links"][:2]
            filtered_metadata["tags"] = filtered_metadata["tags"][:2]
            if logger:
                logger.log_warning("Document metadata size reduced", {
                    "doc_path": metadata.get("source", "unknown"),
                    "original_size": len(metadata_str.encode('utf-8')),
                    "reduced_size": len(str(filtered_metadata).encode('utf-8'))
                })
        
        return Document(page_content=doc.page_content, metadata=filtered_metadata)
    except Exception as e:
        if logger:
            logger.log_error(e, {
                "context": "metadata_enrichment",
                "doc_path": doc.metadata.get("source", "unknown")
            })
        raise

def upsert_obsidian_vault(vault_path: str) -> None:
    """
    1. Loads all markdown from an Obsidian vault
    2. Chunks content preserving header context
    3. Upserts only the changed/new files to Pinecone
    4. Tracks files with 'indexed_files.json' so we only upsert deltas next time
    """
    start_time = time.time()
    success = False
    processed_files = 0
    total_chunks = 0

    try:
        # Initialize Pinecone
        if not PINECONE_API_KEY:
            raise ValueError("Missing Pinecone API key. Please set PINECONE_API_KEY in your .env file.")
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        try:
            # Try to get the index first
            index = pc.Index(INDEX_NAME)
            print(f"Connected to existing index: {INDEX_NAME}")
        except Exception as e:
            # If index doesn't exist, create it
            print(f"Creating new index: {INDEX_NAME}")
            if logger:
                logger.log_warning("Creating new Pinecone index", {"index_name": INDEX_NAME})
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
        is_first_run = len(upsert_records) == 0
        print(f"First run: {is_first_run} (found {len(upsert_records)} existing records)")

        # Load documents from Obsidian
        input_path = Path(vault_path).expanduser().resolve()
        if not input_path.exists():
            raise ValueError(f"Vault path does not exist: {vault_path}")

        print(f"Loading markdown files from: {input_path}")
        
        # First get all markdown files in the vault
        markdown_files = list(input_path.rglob("*.md"))
        print(f"Found {len(markdown_files)} markdown files in vault.")
        
        # Load and chunk each file individually
        chunks_to_upsert = []
        updated_records = dict(upsert_records)  # copy so we don't mutate in-place

        # Check each file for changes
        print("\nProcessing documents...")
        for file_path in markdown_files:
            try:
                # Check if file needs processing
                mtime = file_path.stat().st_mtime
                prev_mtime = upsert_records.get(str(file_path), 0.0)
                
                if not is_first_run and mtime <= prev_mtime:
                    continue  # Skip unchanged files
                    
                # Read and chunk the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Get chunks with header context
                rel_path = file_path.relative_to(input_path)
                chunks = chunk_markdown(content, str(rel_path))
                
                # Enrich metadata for each chunk
                for chunk in chunks:
                    try:
                        enriched_chunk = enrich_obsidian_metadata(chunk)
                        chunks_to_upsert.append(enriched_chunk)
                    except Exception as e:
                        print(f"Error enriching chunk metadata for {file_path}: {str(e)}")
                        if logger:
                            logger.log_error(e, {
                                "context": "enriching_chunk_metadata",
                                "file_path": str(file_path)
                            })
                        continue
                
                updated_records[str(file_path)] = mtime
                total_chunks += len(chunks)
                
                if len(chunks_to_upsert) % 100 == 0:
                    print(f"Processed {len(chunks_to_upsert)} chunks...")
                    if logger:
                        logger.log_performance_metrics(get_performance_metrics())
                        
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                if logger:
                    logger.log_error(e, {
                        "context": "processing_file",
                        "file_path": str(file_path)
                    })
                continue

        # Upsert only if we have deltas
        if not chunks_to_upsert:
            print("No new or changed markdown files to upsert.")
            success = True
            return

        print(f"\nUpserting {len(chunks_to_upsert)} chunks to Pinecone...")
        
        # Batch chunks to avoid rate limits
        batch_size = 20  # Even smaller batch size
        total_batches = (len(chunks_to_upsert) + batch_size - 1) // batch_size
        
        for i in range(0, len(chunks_to_upsert), batch_size):
            batch = chunks_to_upsert[i:i + batch_size]
            current_batch = i // batch_size + 1
            
            print(f"Upserting batch {current_batch} of {total_batches} ({len(batch)} chunks)...")
            try:
                vectorstore.add_documents(batch)
                processed_files += len(batch)
                
                # Log performance metrics every few batches
                if current_batch % 5 == 0:
                    if logger:
                        logger.log_performance_metrics(get_performance_metrics())
                        
            except Exception as e:
                print(f"Error upserting batch {current_batch}: {str(e)}")
                if logger:
                    logger.log_batch_failure(
                        [doc.metadata.get('source', 'unknown') for doc in batch],
                        e
                    )
                continue

        # Save the updated records
        save_indexed_records(updated_records)
        success = True
        
        print(f"\nProcessing complete:")
        print(f"- Total files processed: {len(markdown_files)}")
        print(f"- Total chunks created: {total_chunks}")
        print(f"- Total chunks upserted: {processed_files}")
        
    except Exception as e:
        print(f"Error during indexing: {str(e)}")
        if logger:
            logger.log_error(e, {"context": "indexing_operation"})
        success = False
        
    finally:
        if logger:
            logger.log_end_indexing(processed_files, success)

if __name__ == "__main__":
    upsert_obsidian_vault(OBSIDIAN_PATH)