from typing import Dict, Any, Optional
import logging

# Milvus
from milvus_haystack import MilvusDocumentStore
# Import the MilvusHybridRetriever if you plan to use it directly in the factory for other methods
from milvus_haystack import MilvusHybridRetriever
from haystack.utils import Secret # If you use Secret for tokens/passwords

class DocumentStoreFactory:
    """Factory for creating and managing document store components."""
    
    def __init__(self, vector_db_cfg: Dict[str, Any] = None):
        self.vector_db_cfg = vector_db_cfg or {}
        self._document_store = None
        self._retriever = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_document_store(self) -> MilvusDocumentStore:
        """Creates a Milvus document store configured for hybrid search."""
        if self._document_store is None:
            self.logger.info("Initializing document store for hybrid search...")

            connection_args_dict: Dict[str, Any] = {}
            # log what is the load_type
            if self.vector_db_cfg.get("load_type") == 'milvus-lite':
                db_path = self.vector_db_cfg.get("db_path") # This should be the 'uri' for Milvus Lite
                if not db_path:
                    raise ValueError("Database path (uri for milvus-lite) is not set. Please provide a valid path.")
                connection_args_dict["uri"] = db_path
            else: # Assuming a remote Milvus instance if not milvus-lite
                # For remote, 'uri' usually includes protocol, host, and port
                uri = self.vector_db_cfg.get("uri")
                connection_args_dict["uri"] = uri
                # Add other potential connection args if present in config (e.g., token for Zilliz Cloud)
                if "token" in self.vector_db_cfg:
                    # Ensure token is handled correctly, possibly as Secret
                    # connection_args_dict["token"] = Secret.from_token(self.vector_db_cfg["token"])
                    connection_args_dict["token"] = self.vector_db_cfg["token"]
                if "user" in self.vector_db_cfg:
                    connection_args_dict["user"] = self.vector_db_cfg["user"]
                if "password" in self.vector_db_cfg:
                    connection_args_dict["password"] = self.vector_db_cfg["password"] # Potentially Secret
                if "secure" in self.vector_db_cfg: # For Zilliz Cloud with https
                     connection_args_dict["secure"] = self.vector_db_cfg["secure"]

            collection_name = self.vector_db_cfg.get("collection_name")
            drop_old = self.vector_db_cfg.get("drop_old", False)

            # Field names (can be customized or use MilvusDocumentStore defaults)
            primary_field = self.vector_db_cfg.get("primary_field", "id")
            text_field = self.vector_db_cfg.get("text_field", "text")
            dense_vector_field = self.vector_db_cfg.get("dense_vector_field", "vector") # Name for the dense vector field
            sparse_vector_field = self.vector_db_cfg.get("sparse_vector_field", "sparse_embedding") # Name for sparse

            # The 'dimension' from your config is for your embedders, not directly for MilvusDocumentStore init
            dense_dimension_for_logging = self.vector_db_cfg.get("dimension")

            self.logger.info(
                f"Preparing MilvusDocumentStore with: "
                f"connection_args: { {k: (v if k != 'token' and k != 'password' else '********') for k,v in connection_args_dict.items()} }, "
                f"collection_name: {collection_name}, drop_old: {drop_old}, "
                f"primary_field: {primary_field}, text_field: {text_field}, "
                f"dense_vector_field: {dense_vector_field}, sparse_vector_field: {sparse_vector_field}."
            )
            if dense_dimension_for_logging:
                 self.logger.info(f"(Dense embedding dimension from config: {dense_dimension_for_logging} - Milvus will infer this from data during collection creation).")

            self._document_store = MilvusDocumentStore(
                connection_args=connection_args_dict,
                collection_name=collection_name,
                drop_old=drop_old,
                primary_field=primary_field,
                text_field=text_field,
                vector_field=dense_vector_field,         # Use the configured name for the dense vector field
                sparse_vector_field=sparse_vector_field, # Use the configured name for the sparse vector field
                # You can also pass other relevant MilvusDocumentStore params from your config:
                # consistency_level=self.vector_db_cfg.get("consistency_level", "Session"),
                # index_params=self.vector_db_cfg.get("index_params"),
                # search_params=self.vector_db_cfg.get("search_params"),
                # enable_dynamic_field=self.vector_db_cfg.get("enable_dynamic_field", True),
                # sparse_index_params=self.vector_db_cfg.get("sparse_index_params"),
                # sparse_search_params=self.vector_db_cfg.get("sparse_search_params"),
            )
            self.logger.info("MilvusDocumentStore instance created. Collection schema will be defined/checked on first write if it doesn't exist or if drop_old is True.")
        else:
            self.logger.debug("Returning existing document store instance")

        return self._document_store
    
    def create_retriever(self, document_store: Optional[MilvusDocumentStore] = None, top_k: int = 5) -> MilvusHybridRetriever:
        """Creates a MilvusHybridRetriever or returns the existing instance."""
        # Ensure document_store is initialized first
        doc_store_instance = document_store if document_store is not None else self.create_document_store()

        if self._retriever is None or document_store is not None: # Re-create if new store provided or not yet created
            self.logger.info(f"Creating MilvusHybridRetriever with top_k: {top_k}")
            new_retriever = MilvusHybridRetriever(
                document_store=doc_store_instance,
                top_k=top_k
                # reranker can be added here if needed
            )
            if document_store is not None: # If store was passed, don't save as default
                return new_retriever
            self._retriever = new_retriever
            self.logger.info("MilvusHybridRetriever successfully initialized.")
        else:
            self.logger.debug("Returning existing MilvusHybridRetriever instance.")

        return self._retriever
    
    @property
    def document_store(self) -> Optional[MilvusDocumentStore]:
        """Returns the current document store instance or None if not initialized."""
        return self._document_store
    
    @property
    def retriever(self) -> Optional[MilvusHybridRetriever]:
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