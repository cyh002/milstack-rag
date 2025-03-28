import logging
from haystack.dataclasses import ChatMessage
from pipeline.template_provider import SevenWondersTemplateProvider
from pipeline.component_factory import RAGComponentFactory
from pipeline.pipeline_builder import PipelineBuilder

class RAGPipeline:
    """Manages the Retrieval-Augmented Generation pipeline with conversational memory."""
    
    def __init__(self, model_name: str = "SeaLLMs/SeaLLMs-v3-1.5B-Chat", 
                api_base_url: str = "http://localhost:8000/v1/"):
        self.model_name = model_name
        self.api_base_url = api_base_url
        
        # Initialize factories and providers
        self.component_factory = RAGComponentFactory()
        self.template_provider = SevenWondersTemplateProvider()
        
        # Initialize core components
        self.document_store = self.component_factory.create_document_store()
        self.doc_embedder = self.component_factory.create_document_embedder()
        
        # Create components dictionary for builder - store document_store separately
        self.components = {"document_store": self.document_store}
        
        # Initialize pipeline
        self.pipeline = None
        
    def setup_document_store(self, documents):
        """Initialize document store with provided documents."""
        if not documents:
            raise ValueError("No documents provided")
        self.doc_embedder.warm_up()
        result = self.doc_embedder.run(documents=documents)
        self.document_store.write_documents(result["documents"])
    
    def build_pipeline(self, temperature: float = 0.7, max_tokens: int = 512):
        """Create and configure the RAG pipeline with conversational memory."""
        # Create pipeline builder
        builder = PipelineBuilder(self.component_factory, self.template_provider)
        
        # Add document store to components
        builder.components.update(self.components)
        
        # Build pipeline
        builder.add_components(self.model_name, self.api_base_url, temperature, max_tokens)
        builder.connect_components()
        self.pipeline = builder.build()

        logging.info("RAG pipeline successfully built.")
        logging.info(f"Pipeline: {self.pipeline.draw(path='pipeline.png')}")
        
        return self.pipeline
    
    def query(self, question: str) -> str:
        """Run the pipeline with the provided question and memory."""
        if not self.pipeline:
            raise ValueError("Pipeline not initialized. Call build_pipeline first.")
            
        response = self.pipeline.run({
            "query_rephrase_prompt_builder": {"query": question},
            "prompt_builder": {"question": question},
            "memory_joiner": {"values": [ChatMessage.from_user(question)]}
        }
        , include_outputs_from=["query_rephrase_llm", "llm"]
        )

        # Log the rephrased query for debugging
        rephrased_query = response.get("query_rephrase_llm", {}).get("replies", [""])[0]
        logging.info(f"Rephrased query: {rephrased_query}")
  
        return response["llm"]["replies"][0].text