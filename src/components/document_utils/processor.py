from abc import ABC, abstractmethod
from typing import List, Dict, Any
import json
from haystack import Document
from haystack.components.converters import CSVToDocument, TextFileToDocument, PyPDFToDocument
from datasets import load_dataset

class BaseDocumentProcessor(ABC):
    """Abstract base class for all document processors."""
    
    @abstractmethod
    def process(self, sources: List[str]) -> List[Document]:
        """Process documents from given sources and return Document objects."""
        pass
    
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Return the file extension this processor handles."""
        pass
    
    def filter_sources(self, sources: List[str]) -> List[str]:
        """Filter sources to only include those with the correct extension."""
        return [s for s in sources if s.lower().endswith(self.file_extension)]

class TextProcessor(BaseDocumentProcessor):
    """Processes text files into documents."""
    
    @property
    def file_extension(self) -> str:
        return '.txt'
    
    def process(self, sources: List[str]) -> List[Document]:
        sources = self.filter_sources(sources)
        if not sources:
            return []
            
        converter = TextFileToDocument()
        return converter.run(sources=sources)["documents"]

class PDFProcessor(BaseDocumentProcessor):
    """Processes PDF files into documents."""
    
    @property
    def file_extension(self) -> str:
        return '.pdf'
    
    def process(self, sources: List[str]) -> List[Document]:
        sources = self.filter_sources(sources)
        if not sources:
            return []
            
        converter = PyPDFToDocument()
        return converter.run(sources=sources)["documents"]

class CSVProcessor(BaseDocumentProcessor):
    """Processes CSV files into documents."""
    
    @property
    def file_extension(self) -> str:
        return '.csv'
    
    def process(self, sources: List[str]) -> List[Document]:
        sources = self.filter_sources(sources)
        if not sources:
            return []
            
        converter = CSVToDocument()
        return converter.run(sources=sources)["documents"]

class JSONProcessor(BaseDocumentProcessor):
    """Processes JSON files into documents."""
    
    @property
    def file_extension(self) -> str:
        return '.json'
    
    def process(self, sources: List[str]) -> List[Document]:
        sources = self.filter_sources(sources)
        if not sources:
            return []
            
        documents = []
        for file_path in sources:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                content = json.dumps(data, indent=2)
                documents.append(Document(content=content, meta={"source": file_path}))
            except Exception as e:
                print(f"Error processing JSON file {file_path}: {str(e)}")
        
        return documents

class HuggingFaceProcessor(BaseDocumentProcessor):
    """Processes HuggingFace datasets into documents."""
    
    @property
    def file_extension(self) -> str:
        # Not applicable for HuggingFace datasets
        return ''
    
    def process(self, dataset_names: List[str]) -> List[Document]:
        documents = []
        
        for dataset_name in dataset_names:
            try:
                print(f"Loading Hugging Face dataset: {dataset_name}")
                dataset = load_dataset(dataset_name, split="train")
                
                if len(dataset) == 0:
                    print(f"Warning: Dataset {dataset_name} returned empty results")
                    continue
                    
                # Convert dataset items to Haystack Documents
                dataset_docs = [Document(content=doc["content"], meta=doc.get("meta", {})) 
                             for doc in dataset]
                
                documents.extend(dataset_docs)
                print(f"Loaded {len(dataset_docs)} documents from dataset '{dataset_name}'")
                
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {str(e)}")
        
        return documents