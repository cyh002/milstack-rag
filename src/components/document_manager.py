from typing import List, Dict, Any, Optional
from pathlib import Path
import os

from datasets import load_dataset
from haystack import Pipeline, Document

from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter

from .document_utils.loader import DocumentLoader

class DocumentManager:
    """Handles loading, processing and indexing of documents from various sources."""
    
    def __init__(self, datasets_config: Dict[str, Any] = None):
        self.datasets_config = datasets_config or {}
        self.document_store = None
        self.embedder = None
        self.loader = DocumentLoader(self.datasets_config)
    
    def set_components(self, document_store, embedder):
        """Set the document store and embedder components."""
        self.document_store = document_store
        self.embedder = embedder
    
    def load_documents(self) -> List[Document]:
        """Load all documents without indexing them."""
        return self.loader.load_all_documents()
    