import gradio as gr

class RAGUIInterface:
    """Manages the Gradio user interface for the RAG application."""
    
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.interface = None
        
    def setup_interface(self):
        """Create and configure the Gradio interface."""
        self.interface = gr.Interface(
            fn=self.rag_pipeline.query,
            inputs=gr.components.Textbox(lines=2, placeholder="Enter your question here..."),
            outputs="text",
            title="Seven Wonders RAG",
            description="Ask questions about the Seven Wonders of the World"
        )
        return self.interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        if not self.interface:
            self.setup_interface()
        self.interface.launch(**kwargs)
