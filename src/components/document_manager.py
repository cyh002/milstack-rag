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

class DocumentManager:
    """Handles loading, processing and indexing of documents from various sources."""
    
    def __init__(self, datasets_config: Dict[str, Any] = None):
        self.datasets_config = datasets_config or {}
        self.document_store = None
        self.embedder = None
    
    # Document loading methods (from DocumentLoader)
    def load_documents(self) -> List[Document]:
        """Main method to load all configured documents."""
        documents = []
        
        # Load Hugging Face datasets
        hf_documents = self.load_hf_documents()
        documents.extend(hf_documents)
        
        print(f"Loaded {len(documents)} documents in total")
        return documents
    
    def load_hf_documents(self) -> List[Document]:
        """Load documents from Hugging Face datasets specified in config."""
        all_documents = []
        
        # Get the list of Hugging Face datasets from config
        hf_datasets = self.datasets_config.get("huggingface", [])
        
        for dataset_name in hf_datasets:
            try:
                print(f"Loading Hugging Face dataset: {dataset_name}")
                dataset = load_dataset(dataset_name, split="train")
                
                if len(dataset) == 0:
                    print(f"Warning: Dataset {dataset_name} returned empty results")
                    continue
                    
                # Convert dataset items to Haystack Documents
                documents = [Document(content=doc["content"], meta=doc.get("meta", {})) 
                             for doc in dataset]
                
                all_documents.extend(documents)
                print(f"Loaded {len(documents)} documents from dataset '{dataset_name}'")
                
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {str(e)}")
        
        return all_documents
    
    # Document processing methods (from DocumentProcessor)
    def process_documents(self, file_paths: List[str], chunk_size: int = 500, 
                          chunk_overlap: int = 50) -> int:
        """Process and index documents from various file types."""
        if not self.document_store or not self.embedder:
            raise ValueError("Document store and embedder must be set before processing documents. This should be initialized in RAGPipeline.")
            
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