from typing import Dict, Any
import logging

# Milvus
from milvus_haystack import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever

class DocumentStoreFactory:
    """Factory for creating document store components."""
    
    def __init__(self, vector_db_cfg: Dict[str, Any] = None):
        self.vector_db_cfg = vector_db_cfg or {}
    
    def create_document_store(self) -> MilvusDocumentStore:
        """Creates a Milvus document store.
        
        Args:
            db_path: Path to Milvus database
            collection_name: Name of the collection
            recreate: If True, drop and recreate the collection
        """
        if self.vector_db_cfg.get("load_type") == 'milvus-lite':
            db_path = self.vector_db_cfg.get("db_path")
            if not db_path:
                raise ValueError("Database path is not set. Please provide a valid path.")
        
        collection_name = self.vector_db_cfg.get("collection_name")
        recreate = self.vector_db_cfg.get("recreate")
        dimension = self.vector_db_cfg.get("dimension")
        logging.info(f"Creating Milvus document store with db_path: {db_path}, "
                     f"collection_name: {collection_name}, recreate: {recreate}, "
                     f"dimension: {dimension}")
        
        # Create the document store
        return MilvusDocumentStore(
            connection_args={"uri": db_path},
            collection_name=collection_name,
            drop_old=recreate
        )
    
    def create_retriever(self, document_store):
        """Creates and returns a retriever for the document store."""
        return MilvusEmbeddingRetriever(document_store=document_store, top_k=5)