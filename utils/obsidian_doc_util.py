#!/usr/bin/env python3
"""
obsidian_doc_util.py

A simple utility to load and display metadata for a single Obsidian document
using LangChain's ObsidianLoader.
"""

import os
from langchain_community.document_loaders import ObsidianLoader
from langchain_core.documents import Document
from typing import Optional


def get_single_doc(vault_path: str, file_path: str) -> Optional[Document]:
    """
    Get a single document from Obsidian and print its metadata.
    
    Args:
        vault_path (str): Path to the Obsidian vault
        file_path (str): Path to the specific file within the vault
        
    Returns:
        Optional[Document]: The loaded document or None if not found/error
    """
    try:
        # Get the absolute path to the file
        abs_file_path = os.path.abspath(os.path.join(vault_path, file_path))
        if not os.path.exists(abs_file_path):
            print(f"File not found: {abs_file_path}")
            return None
            
        # Initialize the ObsidianLoader with just the directory containing our target file
        file_dir = os.path.dirname(abs_file_path)
        loader = ObsidianLoader(file_dir)
        
        # Load documents
        docs = loader.load()
        
        print(f"\nLooking for file: {os.path.basename(abs_file_path)}")
        print(f"Found {len(docs)} documents in {file_dir}")
        
        target_doc = None
        target_filename = os.path.basename(abs_file_path)
        
        for doc in docs:
            source = doc.metadata.get("source", "")
            if os.path.basename(source) == target_filename:
                target_doc = doc
                break
            
        if not target_doc:
            print(f"Document not found in loader output: {file_path}")
            print("\nAvailable documents:")
            for doc in docs[:5]:  # Show first 5 docs for debugging
                print(f"- {os.path.basename(doc.metadata.get('source', 'No source'))}")
            if len(docs) > 5:
                print(f"... and {len(docs) - 5} more")
            return None
            
        # Print metadata
        print("\nDocument Metadata:")
        print("-----------------")
        for key, value in target_doc.metadata.items():
            print(f"{key}: {value}")
            
        return target_doc
        
    except Exception as e:
        print(f"Error loading document: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and display metadata for a single Obsidian document")
    parser.add_argument("vault_path", help="Path to the Obsidian vault")
    parser.add_argument("file_path", help="Path to the specific file within the vault")
    
    args = parser.parse_args()
    
    doc = get_single_doc(args.vault_path, args.file_path)
