from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
import json
from pathlib import Path

from haystack import Pipeline, Document
from haystack.components.converters import CSVToDocument, TextFileToDocument, PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from datasets import load_dataset

# Setup logging
logger = logging.getLogger(__name__)

class DocumentIndexer(ABC):
    """Base class for document processing and indexing."""
    
    def __init__(self, document_store, document_embedder, config: Optional[Dict[str, Any]] = None):
        self.document_store = document_store
        self.document_embedder = document_embedder
        self.config = config or {}
        self.pipeline = Pipeline()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.initialized = False
    
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Return the file extension this indexer handles."""
        pass
    
    @abstractmethod
    def setup_pipeline(self) -> None:
        """Configure the pipeline components and connections."""
        pass
    
    def filter_sources(self, sources: List[str]) -> List[str]:
        """Filter sources to only include those with the correct extension."""
        if not self.file_extension:
            return sources
        return [s for s in sources if s.lower().endswith(self.file_extension)]
    
    def index_documents(self, sources: List[str]) -> int:
        """
        Index documents from the provided sources.
        
        Args:
            sources: List of file paths or other source identifiers
            
        Returns:
            int: Number of documents indexed
        """
        if not self.initialized:
            self.setup_pipeline()
            self.initialized = True
            
        if not sources:
            self.logger.warning("No sources provided for indexing")
            return 0
        
        # Filter sources based on file extension
        filtered_sources = self.filter_sources(sources)
        if not filtered_sources:
            self.logger.warning(f"No sources with extension {self.file_extension} found")
            return 0
            
        # Document pipeline entry point may differ by indexer type
        entry_point = self.get_pipeline_entry_point()
        result = self.pipeline.run({entry_point: {"sources": filtered_sources}})
        
        # Return the count of indexed documents
        return len(result.get("writer", {}).get("documents", []))
    
    def get_pipeline_entry_point(self) -> str:
        """Return the name of the first component in the pipeline."""
        return "converter"
    
    def validate(self) -> bool:
        """Validate pipeline configuration."""
        if self.document_store is None:
            self.logger.error("Document store not initialized")
            return False
        if self.document_embedder is None:
            self.logger.error("Document embedder not initialized")
            return False
        return True

    @classmethod
    def load_from_directory(cls, directory: str, document_store, document_embedder, config=None):
        """
        Load and index all documents of the indexer's type from a directory.
        
        Args:
            directory: Directory to scan for documents
            document_store: Document store for indexing
            document_embedder: Embedder for documents
            config: Configuration for the indexer
            
        Returns:
            int: Number of documents indexed
        """
        indexer = cls(document_store, document_embedder, config)
        
        # Find all files with matching extension in the directory
        path = Path(directory)
        if not path.exists():
            indexer.logger.warning(f"Directory {directory} does not exist")
            return 0
            
        files = list(path.glob(f'**/*{indexer.file_extension}'))
        file_paths = [str(f) for f in files]
        
        if not file_paths:
            indexer.logger.warning(f"No {indexer.file_extension} files found in {directory}")
            return 0
            
        indexer.logger.info(f"Found {len(file_paths)} {indexer.file_extension} files in {directory}")
        return indexer.index_documents(file_paths)


class TextIndexer(DocumentIndexer):
    """Handles processing and indexing of text documents."""
    
    @property
    def file_extension(self) -> str:
        return '.txt'
    
    def setup_pipeline(self) -> None:
        # Create components
        self.pipeline.add_component("converter", TextFileToDocument())
        
        # Add document splitter
        split_by = self.config.get("split_by", "sentence")
        split_length = self.config.get("split_length", 2)
        split_overlap = self.config.get("split_overlap", 0)
        self.pipeline.add_component("splitter", DocumentSplitter(
            split_by=split_by, 
            split_length=split_length,
            split_overlap=split_overlap
        ))
        
        # Add embedder and writer
        self.pipeline.add_component("embedder", self.document_embedder)
        self.pipeline.add_component("writer", DocumentWriter(self.document_store))
        
        # Connect components
        self.pipeline.connect("converter", "splitter")
        self.pipeline.connect("splitter", "embedder")
        self.pipeline.connect("embedder", "writer")


class PDFIndexer(DocumentIndexer):
    """Handles processing and indexing of PDF documents."""
    
    @property
    def file_extension(self) -> str:
        return '.pdf'
    
    def setup_pipeline(self) -> None:
        # Create components
        self.pipeline.add_component("converter", PyPDFToDocument())
        
        # Add document splitter
        split_by = self.config.get("split_by", "sentence")
        split_length = self.config.get("split_length", 2)
        split_overlap = self.config.get("split_overlap", 0)
        self.pipeline.add_component("splitter", DocumentSplitter(
            split_by=split_by, 
            split_length=split_length,
            split_overlap=split_overlap
        ))
        
        # Add embedder and writer
        self.pipeline.add_component("embedder", self.document_embedder)
        self.pipeline.add_component("writer", DocumentWriter(self.document_store))
        
        # Connect components
        self.pipeline.connect("converter", "splitter")
        self.pipeline.connect("splitter", "embedder")
        self.pipeline.connect("embedder", "writer")


class CSVIndexer(DocumentIndexer):
    """Handles processing and indexing of CSV documents."""
    
    @property
    def file_extension(self) -> str:
        return '.csv'
    
    def setup_pipeline(self) -> None:
        # Create components
        self.pipeline.add_component("converter", CSVToDocument())
        
        # Add document splitter
        split_by = self.config.get("split_by", "sentence")
        split_length = self.config.get("split_length", 2)
        split_overlap = self.config.get("split_overlap", 0)
        self.pipeline.add_component("splitter", DocumentSplitter(
            split_by=split_by, 
            split_length=split_length,
            split_overlap=split_overlap
        ))
        
        # Add embedder and writer
        self.pipeline.add_component("embedder", self.document_embedder)
        self.pipeline.add_component("writer", DocumentWriter(self.document_store))
        
        # Connect components
        self.pipeline.connect("converter", "splitter")
        self.pipeline.connect("splitter", "embedder")
        self.pipeline.connect("embedder", "writer")


class JSONIndexer(DocumentIndexer):
    """Handles processing and indexing of JSON documents."""
    
    @property
    def file_extension(self) -> str:
        return '.json'
    
    def setup_pipeline(self) -> None:
        # JSON doesn't have a built-in converter in Haystack, so we'll handle it differently
        # We'll use a custom method to load and convert JSON files
        
        # Add document splitter
        split_by = self.config.get("split_by", "sentence")
        split_length = self.config.get("split_length", 2)
        split_overlap = self.config.get("split_overlap", 0)
        self.pipeline.add_component("splitter", DocumentSplitter(
            split_by=split_by, 
            split_length=split_length,
            split_overlap=split_overlap
        ))
        
        # Add embedder and writer
        self.pipeline.add_component("embedder", self.document_embedder)
        self.pipeline.add_component("writer", DocumentWriter(self.document_store))
        
        # Connect components (note: no converter since we're handling JSON manually)
        self.pipeline.connect("splitter", "embedder")
        self.pipeline.connect("embedder", "writer")
    
    def index_documents(self, sources: List[str]) -> int:
        """Override to handle JSON files since they need custom processing."""
        if not self.initialized:
            self.setup_pipeline()
            self.initialized = True
            
        sources = self.filter_sources(sources)
        if not sources:
            self.logger.warning("No JSON sources provided for indexing")
            return 0
        
        # Process JSON files manually
        documents = []
        for file_path in sources:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                content = json.dumps(data, indent=2)
                documents.append(Document(content=content, meta={"source": file_path}))
                self.logger.info(f"Processed JSON file: {file_path}")
            except Exception as e:
                self.logger.error(f"Error processing JSON file {file_path}: {str(e)}")
        
        if not documents:
            return 0
            
        # Process the documents through the pipeline starting at the splitter
        result = self.pipeline.run({"splitter": {"documents": documents}})
        
        # Return the count of indexed documents
        return len(result.get("writer", {}).get("documents", []))


class HuggingFaceIndexer(DocumentIndexer):
    """Handles processing and indexing of HuggingFace datasets."""
    
    @property
    def file_extension(self) -> str:
        # Not applicable for HuggingFace datasets
        return ''
    
    def setup_pipeline(self) -> None:
        # HuggingFace doesn't need a converter
        
        # Add document splitter
        split_by = self.config.get("split_by", "sentence")
        split_length = self.config.get("split_length", 2)
        split_overlap = self.config.get("split_overlap", 0)
        self.pipeline.add_component("splitter", DocumentSplitter(
            split_by=split_by, 
            split_length=split_length,
            split_overlap=split_overlap
        ))
        
        # Add embedder and writer
        self.pipeline.add_component("embedder", self.document_embedder)
        self.pipeline.add_component("writer", DocumentWriter(self.document_store))
        
        # Connect components (note: no converter)
        self.pipeline.connect("splitter", "embedder")
        self.pipeline.connect("embedder", "writer")
    
    def index_documents(self, dataset_names: List[str]) -> int:
        """Override to handle HuggingFace datasets."""
        if not self.initialized:
            self.setup_pipeline()
            self.initialized = True
            
        if not dataset_names:
            self.logger.warning("No dataset names provided for indexing")
            return 0
        
        total_docs = 0
        
        for dataset_name in dataset_names:
            try:
                self.logger.info(f"Loading HuggingFace dataset: {dataset_name}")
                dataset = load_dataset(dataset_name, split="train")
                
                if len(dataset) == 0:
                    self.logger.warning(f"Dataset {dataset_name} returned empty results")
                    continue
                
                # Convert dataset items to Haystack Documents
                documents = []
                for doc in dataset:
                    if "content" in doc:
                        documents.append(Document(
                            content=doc["content"],
                            meta={**doc.get("meta", {}), "source": f"huggingface:{dataset_name}"}
                        ))
                
                if not documents:
                    self.logger.warning(f"No valid documents found in dataset {dataset_name}")
                    continue
                
                # Process documents through the pipeline
                result = self.pipeline.run({"splitter": {"documents": documents}})
                num_indexed = len(result.get("writer", {}).get("documents", []))
                
                self.logger.info(f"Indexed {num_indexed} documents from dataset '{dataset_name}'")
                total_docs += num_indexed
                
            except Exception as e:
                self.logger.error(f"Error processing dataset {dataset_name}: {str(e)}")
        
        return total_docs


class DocumentIndexerFactory:
    """Factory for creating document indexers based on file type."""
    
    INDEXERS = {
        'txt': TextIndexer,
        'pdf': PDFIndexer,
        'csv': CSVIndexer,
        'json': JSONIndexer,
        'huggingface': HuggingFaceIndexer
    }
    
    @classmethod
    def create_indexer(cls, doc_type: str, document_store, document_embedder, config=None):
        """
        Create an indexer for the specified document type.
        
        Args:
            doc_type: Type of document (txt, pdf, csv, json, huggingface)
            document_store: Document store for indexing
            document_embedder: Embedder for documents
            config: Configuration for the indexer
            
        Returns:
            DocumentIndexer: The appropriate indexer instance
        """
        if doc_type not in cls.INDEXERS:
            raise ValueError(f"Unknown document type: {doc_type}. Available types: {list(cls.INDEXERS.keys())}")
            
        return cls.INDEXERS[doc_type](document_store, document_embedder, config)