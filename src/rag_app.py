"""
RAG application using Haystack, vLLM, and Seven Wonders dataset.
"""
from typing import Any, List

from data.document_loader import DocumentLoader
from pipeline.rag_pipeline import RAGPipeline
from ui.gradio_interface import RAGUIInterface

class SevenWondersRAG:
    """Main application class combining all components."""
    
    def __init__(self, model_name="SeaLLMs/SeaLLMs-v3-1.5B-Chat"):
        self.doc_loader = DocumentLoader("bilgeyucel/seven-wonders")
        self.rag_pipeline = RAGPipeline(model_name=model_name)
        self.ui = None
        
    def setup(self):
        """Set up all components of the application."""
        # Load documents
        documents = self.doc_loader.load_documents()
        
        # Setup pipeline
        self.rag_pipeline.setup_document_store(documents)
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