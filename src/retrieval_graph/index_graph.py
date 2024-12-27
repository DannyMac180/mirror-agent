"""This graph exposes endpoints for indexing documents, including Obsidian markdown files."""

from pathlib import Path
from typing import Optional, Sequence, Union

from langchain_community.document_loaders import ObsidianLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from retrieval_graph import retrieval
from retrieval_graph.configuration import IndexConfiguration
from retrieval_graph.state import IndexState


OBSIDIAN_PATH = "/Users/danielmcateer/Library/Mobile Documents/iCloud~md~obsidian/Documents/Ideaverse"

def ensure_docs_have_user_id(
    docs: Sequence[Document], config: RunnableConfig
) -> list[Document]:
    """Ensure that all documents have a user_id in their metadata.

    Args:
        docs (Sequence[Document]): A sequence of Document objects to process.
        config (RunnableConfig): A configuration object containing the user_id.

    Returns:
        list[Document]: A new list of Document objects with updated metadata.
    """
    user_id = config["configurable"]["user_id"]
    return [
        Document(
            page_content=doc.page_content, 
            metadata={
                **doc.metadata, 
                "user_id": user_id,
                "source_type": "obsidian" if doc.metadata.get("source") and ".md" in doc.metadata["source"] else "direct"
            }
        )
        for doc in docs
    ]

def is_obsidian_path(path: str) -> bool:
    """Check if a path is an Obsidian vault path.
    
    Args:
        path (str): Path to check
        
    Returns:
        bool: True if path is an Obsidian vault path
    """
    return OBSIDIAN_PATH in str(Path(path).expanduser().resolve())

def load_obsidian_docs(vault_path: str) -> list[Document]:
    """Load all markdown files from an Obsidian vault.
    
    Args:
        vault_path (str): Path to the Obsidian vault directory
        
    Returns:
        list[Document]: List of documents from markdown files
    """
    # Convert string path to Path object
    vault_dir = Path(vault_path).expanduser().resolve()
    if not vault_dir.exists():
        raise ValueError(f"Obsidian vault path does not exist: {vault_path}")
        
    # Initialize Obsidian loader
    loader = ObsidianLoader(
        str(vault_dir),
        get_attachments=False,  # Skip attachments for now
        collect_metadata=True
    )
    
    print(f"Loading markdown files from Obsidian vault: {vault_dir}")
    
    # Load all markdown documents
    docs = loader.load()
    print(f"Loaded {len(docs)} markdown files from vault")
    return docs

async def index_docs(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    This function handles both direct document input and Obsidian vault paths.
    For Obsidian vaults, it loads all markdown files and indexes them.

    Args:
        state (IndexState): The current state containing documents or vault path.
        config (Optional[RunnableConfig]): Configuration for the indexing process.
    """
    if not config:
        raise ValueError("Configuration required to run index_docs.")

    docs_to_index = []
    
    # Process existing docs in state
    if state.docs:
        for doc in state.docs:
            # Check if this is an Obsidian vault path
            if isinstance(doc.page_content, str) and is_obsidian_path(doc.page_content):
                # Load all markdown files from the vault
                obsidian_docs = load_obsidian_docs(doc.page_content)
                docs_to_index.extend(obsidian_docs)
            else:
                # Regular document
                docs_to_index.append(doc)
    
    # Add user_id and metadata
    stamped_docs = ensure_docs_have_user_id(docs_to_index, config)
    
    print(f"Indexing {len(stamped_docs)} documents...")
    
    # Index documents using retriever
    with retrieval.make_retriever(config) as retriever:
        await retriever.aadd_documents(stamped_docs)
        
    print("Indexing complete!")
    return {"docs": "delete"}


# Define the graph
builder = StateGraph(IndexState, config_schema=IndexConfiguration)
builder.add_node("index_docs", index_docs)
builder.add_edge("__start__", "index_docs")

# Compile the graph
graph = builder.compile()
graph.name = "IndexGraph"
