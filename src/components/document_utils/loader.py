from pathlib import Path
from typing import List, Dict, Any, Optional
from haystack import Document
from .processor import TextProcessor, PDFProcessor, CSVProcessor, JSONProcessor, HuggingFaceProcessor
import logging

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Handles loading documents from different sources."""
    
    def __init__(self, datasets_config: Dict[str, Any]):
        self.datasets_config = datasets_config
        self.processors = {
            'txt': TextProcessor(),
            'pdf': PDFProcessor(),
            'csv': CSVProcessor(),
            'json': JSONProcessor(),
            'huggingface': HuggingFaceProcessor()
        }
        self.validate_processors()
        logger.info("DocumentLoader initialized with processors: %s", list(self.processors.keys()))

    def validate_processors(self):
        """Validate the processors based on the datasets_config."""
        # Get the 'include' value from datasets_config
        included_processors = self.datasets_config.get('include')
        logger.info("Included processors from config: %s", included_processors)
        # If include is None/null (~), keep all processors (default behavior)
        if included_processors is None:
            logger.info("No specific processors included, using all available processors")
            return
        # Check if all specified processors exist
        unknown_processors = [p for p in included_processors if p not in self.processors]
        if unknown_processors:
            raise ValueError(f"Unknown processor types: {unknown_processors}. Available types: {list(self.processors.keys())}")
        
        # Always keep 'huggingface' processor if it has datasets configured
        hf_datasets = self.datasets_config.get('huggingface', [])
        keep_huggingface = len(hf_datasets) > 0
        
        # Filter processors to only include those specified
        self.processors = {k: v for k, v in self.processors.items() 
                        if k in included_processors or (k == 'huggingface' and keep_huggingface)}
        logger.info(f"Using processors: {list(self.processors.keys())}")

    def load_all_documents(self) -> List[Document]:
        """Load all documents from configured sources."""
        documents = []
        
        # Get base directory from config
        base_dir = self.datasets_config.get('dir_key')
        if base_dir and Path(base_dir).exists():
            # Load files directly from base directory
            base_path = Path(base_dir)
            
            # Process each file type
            for processor_key, processor in self.processors.items():
                if processor_key != 'huggingface':  # Skip non-file processors
                    files = list(base_path.glob(f'**/*{processor.file_extension}'))
                    file_paths = [str(f) for f in files]
                    if file_paths:
                        documents.extend(processor.process(file_paths))
                        logger.info(f"Loaded {len(file_paths)} documents from {processor_key} files")
                    else:
                        logger.warning(f"No {processor.file_extension} files found in {base_dir}")
        
        # Load from HuggingFace
        hf_datasets = self.datasets_config.get('huggingface', [])
        if hf_datasets:
            hf_docs = self.processors['huggingface'].process(hf_datasets)
            if hf_docs:
                logger.info(f"Loaded {len(hf_docs)} documents from Hugging Face datasets")
            else:
                logger.warning("No documents loaded from Hugging Face datasets")
            documents.extend(hf_docs)
        
        print(f"Loaded {len(documents)} documents in total")
        return documents
        
