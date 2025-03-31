"""
Main entry point for the RAG application.
"""
from rag_app import MilstackRAG
import logging
from components.config import ConfigLoader
from pathlib import Path
import os
from dotenv import load_dotenv
from components.unified_document_manager import UnifiedDocumentManager
from pipeline.component_factory import RAGComponentFactory

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler("debug.log"), # Log to file
            logging.StreamHandler()  # Log to console
        ]
    )
    logging.info("Logging setup complete.")
    logging.info("Logging to file and console.")

def setup_environment():
    """Set up environment variables."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        # Export loaded variables to actual environment variables for OmegaConf
        for key, value in os.environ.items():
            if key.startswith(('VLLM_', 'OPENAI_', 'MILVUS_', 'SENTENCE_', 'DB_')):
                os.environ[key] = value
        logging.info(f"Loaded environment variables from {env_path}")
    else:
        logging.warning(f".env file not found at {env_path}. Environment variables will not be loaded.")

def setup_config():
    """Set up configuration loader."""
    config_path = Path(__file__).resolve().parent.parent / "conf/config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    config_loader = ConfigLoader(config_path)
    return config_loader

def initialize_document_system(config):
    """Initialize the document processing system."""
    # Create component factory
    component_factory = RAGComponentFactory(config)
    
    # Create document store and embedder
    document_store = component_factory.create_document_store()
    document_embedder = component_factory.create_document_embedder()
    
    # Create document manager
    doc_manager = UnifiedDocumentManager(
        document_store=document_store,
        document_embedder=document_embedder,
        config=config
    )
    
    # Process documents (will either load or index based on config)
    result = doc_manager.process_documents()
    
    if isinstance(result, int):
        logging.info(f"Indexed {result} documents")
    else:
        logging.info(f"Loaded {len(result)} documents without indexing")
        
    return doc_manager, document_store

def test_query(app):
    """Test query function."""
    # Example test query
    question = "What does Rhodes Statue look like?"
    logging.info(f"Question: {question}")
    answer = app.run_query(question)
    logging.info(f"Answer: {answer}")

def main():
    # For command-line usage
    setup_logging()
    setup_environment()
    # Load configuration
    config_loader = setup_config()
    config_loader.log_environment_variables(
        mask_secrets=True,
        prefix_filter=['VLLM_', 'OPENAI_', 'MILVUS_', 'SENTENCE_']
    )
    config = config_loader.get_config()
    logging.info("Configuration loaded successfully.")
    logging.info(f"Configuration: {config}")

    logging.info("Starting Milstack RAG application...")

    # Pass config to MilstackRAG
    app = MilstackRAG(config=config).setup()
    logging.info("Application setup complete.")
    logging.info("Ready to accept queries.")

    # Initialize document system
    doc_manager, document_store = initialize_document_system(config)
    logging.info("Document system initialized successfully.")

    # Example query
    logging.info("Running test query...")
    test_query(app)
    logging.info("Test query successful.")

    logging.info("Launching UI...")
    # Launch UI
    app.launch_ui()

if __name__ == "__main__":
    main()