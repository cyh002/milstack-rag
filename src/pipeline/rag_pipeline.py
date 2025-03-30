import logging
from haystack.dataclasses import ChatMessage
from pipeline.template_provider import MilStackTemplateProvider
from pipeline.component_factory import RAGComponentFactory
from pipeline.pipeline_builder import PipelineBuilder
from components.llm import LLMProvider

class RAGPipeline:
    """Manages the Retrieval-Augmented Generation pipeline with conversational memory."""
    
    def __init__(self, 
                 config=None):
        # Initialize factories and providers
        self.config = config
        self.component_factory = RAGComponentFactory(config=self.config)
        self.template_provider = MilStackTemplateProvider()
        
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
        
        logging.info(f"Embedding {len(documents)} documents...")
        self.doc_embedder.warm_up()
        result = self.doc_embedder.run(documents=documents) 
        logging.info("Writing documents to Milvus document store...")
        self.document_store.write_documents(result["documents"])
        logging.info(f"Successfully stored {len(documents)} documents in Milvus")
    
    def build_pipeline(self, temperature: float = 0.7, max_tokens: int = 512):
        """Create and configure the RAG pipeline with conversational memory."""
        # Create pipeline builder
        builder = PipelineBuilder(self.component_factory, self.template_provider)
        
        # Add document store to components
        builder.components.update(self.components)
        
        # Build pipeline
        # Extract LLM configuration from config
        llm_config = self.config.get("llm", {})
        provider_type = llm_config.get("provider", "local")
        provider_config = llm_config.get(provider_type, {})
        model_name = provider_config.get("model_name")
        api_base_url = provider_config.get("base_url")

        builder.add_components(model_name, api_base_url, temperature, max_tokens)
        builder.connect_components()
        self.pipeline = builder.build()

        logging.info("RAG pipeline successfully built.")
        logging.info(f"Pipeline: {self.pipeline.draw(path='pipeline.png')}")
        
        return self.pipeline
    
    def query(self, question: str) -> str:
        """Run the pipeline with the provided question and memory."""
        if not self.pipeline:
            raise ValueError("Pipeline not initialized. Call build_pipeline first.")
            
        logging.info(f"Running query: {question}")
        
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