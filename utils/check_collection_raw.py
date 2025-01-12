#!/usr/bin/env python3
import chromadb
import numpy as np

# Create a client
client = chromadb.PersistentClient(path="/Users/danielmcateer/Desktop/dev/mirror-agent/data/chroma")

# Get the collection
collection = client.get_collection("obsidian")

# Get raw data
result = collection.get(limit=1, include=['embeddings', 'documents', 'metadatas'])
print("\nRaw result:")
print(f"Keys in result: {result.keys()}")

try:
    if len(result['embeddings']) > 0:
        emb = np.array(result['embeddings'][0])
        print(f"\nEmbedding shape: {emb.shape}")
except Exception as e:
    print(f"\nError getting embeddings: {e}")
    print(f"Type of embeddings: {type(result['embeddings'])}")
    if hasattr(result['embeddings'], 'shape'):
        print(f"Shape: {result['embeddings'].shape}")

try:
    if len(result['documents']) > 0:
        print(f"\nSample document: {result['documents'][0][:200]}")
except Exception as e:
    print(f"\nError getting documents: {e}")

try:
    if len(result['metadatas']) > 0:
        print(f"\nSample metadata: {result['metadatas'][0]}")
except Exception as e:
    print(f"\nError getting metadata: {e}")
