from typing import List, Dict, Optional, Union
import chromadb
from chromadb.api.types import QueryResult

class ChromaUtils:
    def __init__(self, client=None, collection_name: str = "default"):
        """Initialize ChromaUtils with a client and collection name.
        
        Args:
            client: ChromaDB client instance. If None, creates a new persistent client.
            collection_name: Name of the collection to use.
        """
        self.client = client or chromadb.PersistentClient()
        self.collection = self.client.get_or_create_collection(collection_name)

    def query(
        self,
        query_texts: Optional[Union[str, List[str]]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> QueryResult:
        """Query the Chroma collection.
        
        Args:
            query_texts: Text(s) to search for. Can be a single string or list of strings.
            query_embeddings: Pre-computed embeddings to search with.
            n_results: Number of results to return.
            where: Filter conditions for metadata.
            where_document: Filter conditions for document content.
            
        Returns:
            QueryResult containing ids, distances, metadatas, and documents.
        """
        if isinstance(query_texts, str):
            query_texts = [query_texts]

        return self.collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document
        )

    def get_collection_info(self) -> Dict:
        """Get information about the current collection.
        
        Returns:
            Dict containing collection name, count, and metadata.
        """
        return {
            "name": self.collection.name,
            "count": self.collection.count(),
            "metadata": self.collection.metadata
        }
