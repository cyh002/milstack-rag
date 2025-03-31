from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from haystack import Document

from .document_utils.indexer import DocumentIndexerFactory
from .document_utils.loader import DocumentLoader

logger = logging.getLogger(__name__)

class UnifiedDocumentManager:
    """
    Unified manager for document loading, processing, and indexing.
    Provides backward compatibility with DocumentManager while adding indexing capabilities.
    """
    
    def __init__(self, document_store=None, document_embedder=None, config: Dict[str, Any] = None):
        self.config = config or {}
        self.datasets_config = self.config.get("datasets", {})
        self.indexing_config = self.config.get("indexing", {})
        self.document_store = document_store
        self.document_embedder = document_embedder
        self.indexers = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize loader for backward compatibility
        self.loader = DocumentLoader(self.datasets_config)
        
        # Check if indexing is enabled
        self.indexing_enabled = self.indexing_config.get("enabled", True)
        self.logger.info(f"Indexing {'enabled' if self.indexing_enabled else 'disabled'}")
    
    def set_components(self, document_store, document_embedder):
        """Set or update the document store and embedder components."""
        self.document_store = document_store
        self.document_embedder = document_embedder
    
    def load_documents(self) -> List[Document]:
        """Load all documents without indexing them (backward compatibility)."""
        return self.loader.load_all_documents()
    
    def process_documents(self, toggle_indexing: bool = None) -> List[Document] | int:
        """
        Process documents - either just load them or load and index them based on configuration.
        
        Args:
            toggle_indexing: Override the config setting for indexing
            
        Returns:
            Either a list of Document objects (if indexing is disabled) or
            the number of documents indexed (if indexing is enabled)
        """
        # Determine whether to index based on parameter or config
        should_index = toggle_indexing if toggle_indexing is not None else self.indexing_enabled
        
        if should_index:
            if not self.document_store or not self.document_embedder:
                self.logger.error("Document store and embedder must be set before indexing")
                raise ValueError("Document store and embedder must be set before indexing")
            return self.index_all_documents()
        else:
            return self.load_documents()
    
    def index_all_documents(self) -> int:
        """Index all documents from configured sources."""
        if not self.document_store or not self.document_embedder:
            raise ValueError("Document store and embedder must be set before indexing")
            
        total_indexed = 0
        
        # Process files from base directory
        base_dir = self.datasets_config.get('dir_key')
        if base_dir and Path(base_dir).exists():
            base_path = Path(base_dir)
            
            # Configure which document types to include
            included_types = self.datasets_config.get('include')
            doc_types = included_types if included_types else ['txt', 'pdf', 'csv', 'json']
            
            for doc_type in doc_types:
                try:
                    indexer = self._get_indexer(doc_type)
                    
                    # Get the file extension from the indexer
                    ext = indexer.file_extension
                    if not ext:  # Skip non-file types like huggingface
                        continue
                        
                    # Get all files with this extension
                    files = list(base_path.glob(f'**/*{ext}'))
                    file_paths = [str(f) for f in files]
                    
                    if file_paths:
                        count = indexer.index_documents(file_paths)
                        self.logger.info(f"Indexed {count} documents from {len(file_paths)} {ext} files")
                        total_indexed += count
                except ValueError as e:
                    self.logger.warning(f"Error with document type {doc_type}: {str(e)}")
        
        # Process HuggingFace datasets
        hf_datasets = self.datasets_config.get('huggingface', [])
        if hf_datasets:
            try:
                hf_indexer = self._get_indexer('huggingface')
                count = hf_indexer.index_documents(hf_datasets)
                self.logger.info(f"Indexed {count} documents from HuggingFace datasets")
                total_indexed += count
            except Exception as e:
                self.logger.error(f"Error indexing HuggingFace datasets: {str(e)}")
        
        self.logger.info(f"Indexed {total_indexed} documents in total")
        return total_indexed
    
    def _get_indexer(self, doc_type: str):
        """Get or create an indexer for the specified document type."""
        if doc_type not in self.indexers:
            self.indexers[doc_type] = DocumentIndexerFactory.create_indexer(
                doc_type, 
                self.document_store, 
                self.document_embedder,
                self.indexing_config
            )
        return self.indexers[doc_type]
    
    # Alias for backward compatibility
DocumentManager = UnifiedDocumentManager