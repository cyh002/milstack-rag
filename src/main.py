"""
Main entry point for the RAG application.
"""
from rag_app import SevenWondersRAG
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

def main():
    # For command-line usage
    app = SevenWondersRAG().setup()
    
    # Example query
    question = "What does Rhodes Statue look like?"
    answer = app.run_query(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    # Launch UI
    app.launch_ui()

if __name__ == "__main__":
    main()