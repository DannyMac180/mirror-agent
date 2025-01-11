from utils.chroma_utils import ChromaUtils

def rename_collection(old_name: str, new_name: str):
    # Initialize ChromaUtils with old and new collection names
    old_cu = ChromaUtils(collection_name=old_name)
    new_cu = ChromaUtils(collection_name=new_name)
    
    # Get all data from old collection
    data = old_cu.collection.get()
    
    # Add data to new collection if there are documents
    if data['documents']:
        new_cu.collection.add(
            documents=data['documents'],
            metadatas=data['metadatas'],
            ids=data['ids']
        )
    
    # Delete old collection
    old_cu.client.delete_collection(old_name)
    print(f'Collection renamed from {old_name} to {new_name}')

if __name__ == "__main__":
    rename_collection('langchain', 'obsidian')
