"""
Main entry point for the RAG application.
"""
from rag_app import MilstackRAG
import logging
from components.config import ConfigLoader
from pathlib import Path
import os
from dotenv import load_dotenv

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

    # Example query
    logging.info("Running test query...")
    test_query(app)
    logging.info("Test query successful.")

    logging.info("Launching UI...")
    # Launch UI
    app.launch_ui()

if __name__ == "__main__":
    main()