"""
RAG application using Haystack, vLLM, and Seven Wonders dataset.
Also serves as the main entry point.
"""
import logging
from pathlib import Path
import os
from dotenv import load_dotenv
from components.config import ConfigLoader

from components.document_manager import DocumentManager
from pipeline.rag_pipeline import RAGPipeline
from ui.gradio_interface import RAGUIInterface

class MilstackRAG:
    """Main application class combining all components."""
    
    def __init__(self, config=None):
        self.config = config
        self.document_manager = DocumentManager(
            datasets_config=self.config.get("datasets"))
        self.rag_pipeline = RAGPipeline(config=self.config)
        self.ui = None
        
    def setup(self):
        """Set up all components of the application."""
        # Load documents
        documents = self.document_manager.load_documents()
        
        # Setup pipeline
        self.rag_pipeline.index_documents(documents)
        self.rag_pipeline.build_pipeline()  
        
        # Create UI
        self.ui = RAGUIInterface(self.rag_pipeline)
        return self
    
    def run_query(self, question):
        """Run a single query through the pipeline."""
        return self.rag_pipeline.query(question)
    
    def launch_ui(self, **kwargs):
        """Launch the user interface."""
        if not self.ui:
            self.ui = RAGUIInterface(self.rag_pipeline)
        self.ui.launch(**kwargs)

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging setup complete.")

def setup_environment():
    """Set up environment variables."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
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
    return ConfigLoader(config_path)

def main():
    setup_logging()
    setup_environment()
    
    config_loader = setup_config()
    config_loader.log_environment_variables(
        mask_secrets=True,
        prefix_filter=['VLLM_', 'OPENAI_', 'MILVUS_', 'SENTENCE_']
    )
    config = config_loader.get_config()
    logging.info("Configuration loaded successfully.")
    logging.info(f"Configuration: {config}")

    logging.info("Starting Milstack RAG application...")
    app = MilstackRAG(config=config).setup()
    logging.info("Application setup complete. Ready to accept queries.")

    # Example test query
    question = "What does Rhodes Statue look like?"
    logging.info(f"Test query: {question}")
    answer = app.run_query(question)
    logging.info(f"Test query result: {answer}")

    logging.info("Launching UI...")
    app.launch_ui()

if __name__ == "__main__":
    main()