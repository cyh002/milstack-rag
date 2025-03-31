from typing import List, Dict, Any, Optional
from pathlib import Path
import os

from datasets import load_dataset
from haystack import Pipeline, Document
from haystack.components.converters import (
    TextFileToDocument,
    PyPDFToDocument,
    CSVToDocument
)
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
    
    def load_and_index_documents(self, chunk_size: int = 500, chunk_overlap: int = 50) -> int:
        """Load all documents, process them, and index them in the document store."""
        if not self.document_store or not self.embedder:
            raise ValueError("Document store and embedder must be set before processing documents.")
        
        # Load all documents
        documents = self.loader.load_all_documents()
        if not documents:
            print("No documents found to process.")
            return 0
        
        # Split documents into chunks
        splitter = DocumentSplitter(
            split_by="word", 
            split_length=chunk_size,
            split_overlap=chunk_overlap
        )
        split_docs = splitter.run(documents=documents)["documents"]
        
        # Embed documents
        embedded_docs = self.embedder.run(documents=split_docs)["documents"]
        
        # Index documents
        self.document_store.write_documents(embedded_docs)
        
        print(f"Processed and indexed {len(embedded_docs)} document chunks")
        return len(embedded_docs)
    
    