"""Local utility to load and format Obsidian documents for indexing."""

import json
from pathlib import Path

from langchain_community.document_loaders import ObsidianLoader
from langchain_core.documents import Document


OBSIDIAN_PATH = "/Users/danielmcateer/Library/Mobile Documents/iCloud~md~obsidian/Documents/Ideaverse"

def load_obsidian_docs(vault_path: str) -> list[Document]:
    """Load all markdown files from an Obsidian vault or subdirectory.
    
    Args:
        vault_path (str): Path to the Obsidian vault directory or subdirectory
        
    Returns:
        list[Document]: List of documents from markdown files
    """
    input_path = Path(vault_path).expanduser().resolve()
    if not input_path.exists():
        raise ValueError(f"Path does not exist: {vault_path}")
        
    # Initialize Obsidian loader
    loader = ObsidianLoader(
        str(input_path),
        collect_metadata=True,  # Get all Obsidian metadata including tags, links, etc.
    )
    
    print(f"Loading markdown files from: {input_path}")
    
    # Load all markdown documents
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")
    
    # Print sample of metadata for verification
    if docs:
        print("\nSample document metadata:")
        print(f"Title: {docs[0].metadata.get('title', 'No title')}")
        print(f"Source: {docs[0].metadata.get('source', 'No source')}")
        print(f"Tags: {docs[0].metadata.get('tags', [])}")
        
    return docs

def format_for_indexer(docs: list[Document]) -> str:
    """Format documents into JSON for the indexer API.
    
    Args:
        docs (list[Document]): List of documents to format
        
    Returns:
        str: JSON string ready for the indexer
    """
    formatted_docs = []
    for doc in docs:
        # Create metadata with source path and other Obsidian metadata
        metadata = {
            "source": doc.metadata.get("source", ""),
            "title": doc.metadata.get("title", ""),
            "tags": doc.metadata.get("tags", []),
            "links": doc.metadata.get("links", []),
            "obsidian_url": f"obsidian://open?vault=Ideaverse&file={doc.metadata.get('source', '').replace(OBSIDIAN_PATH + '/', '')}"
        }
        
        # Format as Document for indexer
        formatted_docs.append({
            "page_content": doc.page_content,
            "metadata": metadata
        })
    
    return json.dumps(formatted_docs, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and format Obsidian documents for indexing")
    parser.add_argument("path", help="Path to Obsidian vault or subdirectory")
    parser.add_argument("--output", "-o", help="Output file for formatted JSON", default="obsidian_docs.json")
    args = parser.parse_args()
    
    try:
        # Load docs
        docs = load_obsidian_docs(args.path)
        
        # Format and save
        formatted = format_for_indexer(docs)
        with open(args.output, "w") as f:
            f.write(formatted)
        
        print(f"\nFormatted documents saved to {args.output}")
        print("You can now copy the contents of this file into the LangGraph Studio UI")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nIf you're seeing metadata-related errors, try running:")
        print("pip install --upgrade langchain-community") 