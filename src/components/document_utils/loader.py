import logging
from pathlib import Path
from typing import List, Dict, Any
from haystack import Document
from .processor import TextProcessor, PDFProcessor, CSVProcessor, JSONProcessor, HuggingFaceProcessor

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
        """Validate and optionally filter processors based on the datasets_config."""
        included_processors = self.datasets_config.get('include')
        logger.info("Included processors from config: %s", included_processors)
        if included_processors is not None:
            unknown = [p for p in included_processors if p not in self.processors]
            if unknown:
                raise ValueError(f"Unknown processor types: {unknown}. Available types: {list(self.processors.keys())}")
            # Filter processors to include only those specified
            self.processors = {k: v for k, v in self.processors.items() if k in included_processors}
            logger.info("Filtered processors: %s", list(self.processors.keys()))
        else:
            logger.info("No specific processors included, using all available processors")

    def load_all_documents(self) -> List[Document]:
        """Load all documents from configured sources."""
        documents = []
        documents.extend(self._load_file_documents())
        documents.extend(self._load_non_file_documents())
        logger.info(f"Total loaded documents: {len(documents)}")
        return documents

    def _load_file_documents(self) -> List[Document]:
        """Load documents from file-based sources."""
        docs: List[Document] = []
        base_dir = self.datasets_config.get('dir_key')
        if not base_dir:
            logger.warning("Directory key 'dir_key' not provided in config, skipping file-based loaders")
            return docs
        base_path = Path(base_dir)
        if not base_path.exists():
            logger.warning(f"Directory '{base_dir}' does not exist, skipping file-based loaders")
            return docs

        for key, processor in self.processors.items():
            if processor.file_extension:
                files = list(base_path.glob(f'**/*{processor.file_extension}'))
                file_paths = [str(f) for f in files]
                if file_paths:
                    proc_docs = processor.process(file_paths)
                    docs.extend(proc_docs)
                    logger.info(f"Loaded {len(proc_docs)} documents from {key} files")
                else:
                    logger.warning(f"No {processor.file_extension} files found in {base_dir} for processor '{key}'")
        return docs

    def _load_non_file_documents(self) -> List[Document]:
        """Load documents from non file-based sources."""
        docs: List[Document] = []
        for key, processor in self.processors.items():
            if not processor.file_extension:  # Non-file-based processors (e.g., HuggingFace)
                sources = self.datasets_config.get(key, [])
                if sources:
                    proc_docs = processor.process(sources)
                    docs.extend(proc_docs)
                    if proc_docs:
                        logger.info(f"Loaded {len(proc_docs)} documents from {key} sources")
                    else:
                        logger.warning(f"No documents loaded from {key} sources")
                else:
                    logger.warning(f"No configuration found for processor '{key}'")
        return docs