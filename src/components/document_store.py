from typing import Dict, Any, Optional
import logging

# Milvus
from milvus_haystack import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever

class DocumentStoreFactory:
    """Factory for creating and managing document store components."""
    
    def __init__(self, vector_db_cfg: Dict[str, Any] = None):
        self.vector_db_cfg = vector_db_cfg or {}
        self._document_store = None
        self._retriever = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_document_store(self) -> MilvusDocumentStore:
        """Creates a Milvus document store if it doesn't exist or returns the existing instance."""
        if self._document_store is None:
            self.logger.info("Initializing document store...")
            
            if self.vector_db_cfg.get("load_type") == 'milvus-lite':
                db_path = self.vector_db_cfg.get("db_path")
                if not db_path:
                    raise ValueError("Database path is not set. Please provide a valid path.")
            
            collection_name = self.vector_db_cfg.get("collection_name")
            recreate = self.vector_db_cfg.get("recreate")
            dimension = self.vector_db_cfg.get("dimension")
            
            self.logger.info(f"Creating Milvus document store with db_path: {db_path}, "
                         f"collection_name: {collection_name}, recreate: {recreate}, "
                         f"dimension: {dimension}")
            
            # Create the document store
            self._document_store = MilvusDocumentStore(
                connection_args={"uri": db_path},
                collection_name=collection_name,
                drop_old=recreate
            )
            self.logger.info("Document store successfully initialized")
        else:
            self.logger.debug("Returning existing document store instance")
            
        return self._document_store
    
    def create_retriever(self, document_store: Optional[MilvusDocumentStore] = None) -> MilvusEmbeddingRetriever:
        """Creates a retriever for the document store or returns the existing instance."""
        if document_store is not None:
            # If document store is explicitly provided, create a new retriever for it
            return MilvusEmbeddingRetriever(document_store=document_store, top_k=5)
            
        if self._retriever is None:
            # Get or create document store
            doc_store = self.create_document_store()
            
            # Create retriever
            self._retriever = MilvusEmbeddingRetriever(document_store=doc_store, top_k=5)
            self.logger.info("Retriever successfully initialized")
        else:
            self.logger.debug("Returning existing retriever instance")
            
        return self._retriever
    
    @property
    def document_store(self) -> Optional[MilvusDocumentStore]:
        """Returns the current document store instance or None if not initialized."""
        return self._document_store
    
    @property
    def retriever(self) -> Optional[MilvusEmbeddingRetriever]:
        """Returns the current retriever instance or None if not initialized."""
        return self._retriever
    
    def initialize(self) -> bool:
        """Initialize both document store and retriever."""
        try:
            self.create_document_store()
            self.create_retriever()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize document store and retriever: {str(e)}")
            return False