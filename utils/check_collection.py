#!/usr/bin/env python3
from chroma_utils import ChromaUtils

# Initialize ChromaUtils with the obsidian collection
chroma = ChromaUtils(collection_name="obsidian")

# Get collection info
info = chroma.get_collection_info()
print(f"Collection Info: {info}")

# Get a sample embedding to check dimensions
result = chroma.collection.get(limit=1)
if result and result['embeddings']:
    dim = len(result['embeddings'][0])
    print(f"\nEmbedding dimensionality: {dim}")
else:
    print("\nNo embeddings found in collection")
