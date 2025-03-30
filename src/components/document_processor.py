from typing import List, Optional
from pathlib import Path
import os

from haystack import Pipeline
from haystack.components.converters import (
    TextFileToDocument,
    PyPDFToDocument,
    CSVToDocument
)
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document

class DocumentProcessor:
    """Processes documents of various file types and indexes them in the document store."""
    
    def __init__(self, document_store, embedder):
        self.document_store = document_store
        self.embedder = embedder
        
    def process_documents(self, file_paths: List[str], chunk_size: int = 500, 
                          chunk_overlap: int = 50) -> None:
        """Process and index documents from various file types."""
        
        # Group files by extension
        txt_files = [f for f in file_paths if f.lower().endswith('.txt')]
        pdf_files = [f for f in file_paths if f.lower().endswith('.pdf')]
        csv_files = [f for f in file_paths if f.lower().endswith('.csv')]
        
        documents = []
        
        # Process each file type with appropriate converter
        if txt_files:
            documents.extend(self._process_text_files(txt_files))
        
        if pdf_files:
            documents.extend(self._process_pdf_files(pdf_files))
            
        if csv_files:
            documents.extend(self._process_csv_files(csv_files))
        
        # Split documents
        if documents:
            splitter = DocumentSplitter(
                split_by="word", 
                split_length=chunk_size,
                split_overlap=chunk_overlap
            )
            split_docs = splitter.run(documents=documents)["documents"]
            
            # Embed and index documents
            embedded_docs = self.embedder.run(documents=split_docs)["documents"]
            self.document_store.write_documents(embedded_docs)
            
            return len(embedded_docs)
        
        return 0
    
    def _process_text_files(self, file_paths: List[str]) -> List[Document]:
        converter = TextFileToDocument()
        return converter.run(sources=file_paths)["documents"]
    
    def _process_pdf_files(self, file_paths: List[str]) -> List[Document]:
        converter = PyPDFToDocument()
        return converter.run(sources=file_paths)["documents"]
    
    def _process_csv_files(self, file_paths: List[str]) -> List[Document]:
        converter = CSVToDocument()
        return converter.run(sources=file_paths)["documents"]