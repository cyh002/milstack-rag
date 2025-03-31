import logging
from haystack.dataclasses import ChatMessage
from haystack import Pipeline as HaystackPipeline

from pipeline.base_pipeline import BasePipeline
from pipeline.template_provider import MilStackTemplateProvider
from pipeline.component_factory import RAGComponentFactory
from pipeline.pipeline_builder import PipelineBuilder
from components.llm import LLMProvider

class RAGPipeline(BasePipeline):
    """Manages the Retrieval-Augmented Generation pipeline with conversational memory."""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Initialize factories and providers
        self.component_factory = RAGComponentFactory(config=self.config)
        self.template_provider = MilStackTemplateProvider()
        
        # Pipeline will be initialized later
        self.haystack_pipeline = None
        self.document_store = None
        self.doc_embedder = None
    
    def initialize(self) -> bool:
        """Initialize all components needed for the pipeline."""
        # Initialize core components
        self.document_store = self.component_factory.create_document_store()
        self.doc_embedder = self.component_factory.create_document_embedder()
        
        # Create components dictionary for builder
        self.components["document_store"] = self.document_store
        
        self.initialized = True
        self.logger.info("RAG Pipeline components initialized")
        return True
        
    def setup_document_store(self, documents):
        """Initialize document store with provided documents."""
        if not self.initialized:
            self.initialize()
            
        if not documents:
            raise ValueError("No documents provided")
        
        self.logger.info(f"Embedding {len(documents)} documents...")
        self.doc_embedder.warm_up()
        result = self.doc_embedder.run(documents=documents) 
        self.logger.info("Writing documents to Milvus document store...")
        self.document_store.write_documents(result["documents"])
        self.logger.info(f"Successfully stored {len(documents)} documents in Milvus")
    
    def build_pipeline(self, temperature: float = 0.7, max_tokens: int = 512):
        """Create and configure the RAG pipeline with conversational memory."""
        if not self.initialized:
            self.initialize()
            
        # Create pipeline builder
        builder = PipelineBuilder(self.component_factory, self.template_provider)
        
        # Add document store to components
        builder.components.update(self.components)
        
        # Extract LLM configuration from config
        llm_config = self.config.get("llm", {})
        provider_type = llm_config.get("provider", "local")
        provider_config = llm_config.get(provider_type, {})
        model_name = provider_config.get("model_name")
        api_base_url = provider_config.get("base_url")

        builder.add_components(model_name, api_base_url, temperature, max_tokens)
        builder.connect_components()
        self.haystack_pipeline = builder.build()

        self.logger.info("RAG pipeline successfully built")
        self.haystack_pipeline.draw(path='pipeline.png')
        self.logger.info(f"Pipeline diagram saved. File path: {self.config.get('pipeline_image_path', 'pipeline.png')}")
        
        return self.haystack_pipeline
    
    def execute(self, question: str, **kwargs) -> str:
        """Execute the RAG pipeline with the provided question.
        This implements the abstract execute() method from BasePipeline.
        """
        return self.query(question)
    
    def query(self, question: str) -> str:
        """Run the pipeline with the provided question and memory."""
        if not self.validate():
            self.initialize()
            self.build_pipeline()
            
        self.logger.info(f"Running query: {question}")
        
        response = self.haystack_pipeline.run({
            "query_rephrase_prompt_builder": {"query": question},
            "prompt_builder": {"question": question},
            "memory_joiner": {"values": [ChatMessage.from_user(question)]}
        }, include_outputs_from=["query_rephrase_llm", "llm"])

        # Log the rephrased query for debugging
        rephrased_query = response.get("query_rephrase_llm", {}).get("replies", [""])[0]
        self.logger.info(f"Rephrased query: {rephrased_query}")

        return response["llm"]["replies"][0].text