import logging
from haystack.dataclasses import ChatMessage
from haystack import Pipeline as HaystackPipeline

from pipeline.base_pipeline import BasePipeline
from pipeline.template_provider import MilStackTemplateProvider
from pipeline.component_factory import RAGComponentFactory
from pipeline.pipeline_builder import PipelineBuilder
from components.llm import LLMProvider
from haystack import Document as HaystackDocument # Alias to avoid confusion if you have other Document classes
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from milvus_haystack import MilvusDocumentStore
from typing import Optional, Any, Dict, List
from milvus_haystack.milvus_embedding_retriever import MilvusHybridRetriever


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
        self.dense_doc_embedder = None
        self.sparse_doc_embedder = None
    
    def initialize(self) -> bool:
        """Initialize all components needed for the pipeline."""
        # Initialize core components
        self.document_store = self.component_factory.create_document_store()
        self.dense_doc_embedder = self.component_factory.create_dense_document_embedder()
        self.sparse_doc_embedder = self.component_factory.create_sparse_document_embedder()
        
        # Create components dictionary for builder
        self.components["document_store"] = self.document_store
        
        self.initialized = True
        self.logger.info("RAG Pipeline core components (doc store, index embedders) initialized")
        return True
    
    def index_documents(self, documents: List[HaystackDocument], policy: DuplicatePolicy = DuplicatePolicy.OVERWRITE):
        """
        Creates embeddings for documents and writes them to the Milvus store.
        This method sets up and runs a dedicated Haystack indexing pipeline.
        """
        if not self.initialized:
            self.initialize()

        if not documents:
            self.logger.warning("No documents provided for indexing.")
            return

        if not self.document_store or not self.dense_doc_embedder or not self.sparse_doc_embedder:
            self.logger.error("Document store or document embedders not initialized. Cannot index.")
            raise RuntimeError("Indexing components not ready.")

        self.logger.info(f"Starting indexing for {len(documents)} documents...")

        # 1. Build the Indexing Pipeline
        indexing_pipeline = HaystackPipeline()
        doc_writer = DocumentWriter(document_store=self.document_store, policy=policy)

        indexing_pipeline.add_component("dense_embedder", self.dense_doc_embedder)
        indexing_pipeline.add_component("sparse_embedder", self.sparse_doc_embedder)
        indexing_pipeline.add_component("writer", doc_writer)

        # Connect components for indexing:
        # Documents go to dense_embedder, then its output (docs with dense embeddings)
        # goes to sparse_embedder, then its output (docs with both embeddings) goes to writer.
        indexing_pipeline.connect("dense_embedder.documents", "sparse_embedder.documents")
        indexing_pipeline.connect("sparse_embedder.documents", "writer.documents")

        self.logger.info("Indexing pipeline built. Running document embedding and writing...")
        try:
            # Warm up embedders if they have such a method (SentenceTransformersEmbedder does)
            if hasattr(self.dense_doc_embedder, 'warm_up'):
                self.dense_doc_embedder.warm_up()
            if hasattr(self.sparse_doc_embedder, 'warm_up'):
                self.sparse_doc_embedder.warm_up()

            # Run the indexing pipeline
            # The initial 'documents' are fed to the first component in the chain ('dense_embedder')
            indexing_pipeline.run({"dense_embedder": {"documents": documents}})
            self.logger.info(f"Successfully indexed {len(documents)} documents into Milvus with dense and sparse embeddings.")
        except Exception as e:
            self.logger.error(f"Error during document indexing: {e}", exc_info=True)
            raise

    # def setup_document_store(self, documents):
    #     """Initialize document store with provided documents."""
    #     if not self.initialized:
    #         self.initialize()
            
    #     if not documents:
    #         raise ValueError("No documents provided")
        
    #     self.logger.info(f"Embedding {len(documents)} documents...")
    #     self.doc_embedder.warm_up()
    #     result = self.doc_embedder.run(documents=documents) 
    #     self.logger.info("Writing documents to Milvus document store...")
    #     self.document_store.write_documents(result["documents"])
    #     self.logger.info(f"Successfully stored {len(documents)} documents in Milvus")
    
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

        if not model_name:
            self.logger.warning("LLM model_name not found in config, LLM creation might fail or use defaults.")

        builder.add_components(model_name, api_base_url, temperature, max_tokens)
        builder.connect_components()
        self.haystack_pipeline = builder.build()

        self.logger.info("RAG pipeline successfully built")
        try:
                    image_path = self.config.get('pipeline', {}).get('image_path', 'rag_query_pipeline.png')
                    self.haystack_pipeline.draw(path=image_path)
                    self.logger.info(f"Query pipeline diagram saved to: {image_path}")
        except Exception as e:
                    self.logger.warning(f"Could not draw pipeline diagram: {e}")
        return self.haystack_pipeline
    
    def execute(self, question: str, **kwargs) -> str:
        """Execute the RAG pipeline with the provided question.
        This implements the abstract execute() method from BasePipeline.
        """
        return self.query(question)
    
    def query(self, question: str) -> str:
        """Run the pipeline with the provided question and memory."""
        if not self.haystack_pipeline: # Check if query pipeline is built
            self.logger.info("Query pipeline not built. Building now...")
            if not self.initialized: # Ensure components like doc store are ready
                self.initialize()
            self.build_pipeline() # Use default temp/tokens or get from config

        self.logger.info(f"Original question for RAG query: {question}") # Log original question

        pipeline_input = {
            "query_rephrase_prompt_builder": {"query": question},
            "prompt_builder": {"question": question},
            "memory_joiner": {"values": [ChatMessage.from_user(question)]}
        }

        # The following part is crucial for hybrid search:
        # The rephrased query from "list_to_str_adapter" will feed into
        # "dense_text_embedder" and "sparse_text_embedder".
        # Their outputs will go to "retriever.query_embedding" and "retriever.query_sparse_embedding".
        # So, the `pipeline_input` primarily needs to feed the start of these chains.
        # If your `query_rephrase_prompt_builder` is the very first component that takes the raw user query
        # (and it seems to be, along with `memory_joiner`), then the `pipeline_input` is likely correct.

        # The `text_embedder` and `sparse_text_embedder` inputs are not directly set here
        # because they are connected to `list_to_str_adapter.output`.
        # However, the original examples for hybrid retrieval sometimes show explicitly passing
        # the text to both embedders in the `run` call if they are entry points.
        # Given your PipelineBuilder's `connect_components`:
        # self.pipeline.connect("list_to_str_adapter.output", "dense_text_embedder.text")
        # self.pipeline.connect("list_to_str_adapter.output", "sparse_text_embedder.text")
        # This means `list_to_str_adapter` (which gets its input from `query_rephrase_llm`)
        # will provide the text to both embedders.
        # Your `pipeline_input` for `query_rephrase_prompt_builder` initiates this chain.

        self.logger.info(f"Running RAG query pipeline with input: { {k:v for k,v in pipeline_input.items() if k != 'memory_joiner'} }...") # Avoid logging full chat messages if too verbose

        response = self.haystack_pipeline.run(
            data=pipeline_input,
            include_outputs_from=["query_rephrase_llm", "llm", "retriever"] # Make sure "retriever" is listed
        )

        rephrased_query_list = response.get("query_rephrase_llm", {}).get("replies", [])
        # log the rephrased query
        self.logger.info(f"Rephrased query list: {rephrased_query_list}")
        rephrased_query = rephrased_query_list[0].text if rephrased_query_list else "N/A"
        # If replies are ChatMessage objects:
        # rephrased_query = rephrased_query_list[0].content if rephrased_query_list and hasattr(rephrased_query_list[0], 'content') else "N/A"

        self.logger.info(f"Rephrased query used for retrieval: {rephrased_query}")
        llm_replies = response.get("llm", {}).get("replies", [])
        # log the LLM replies
        self.logger.info(f"LLM replies: {llm_replies}")
        final_answer = llm_replies[0].text if llm_replies else "No answer from LLM."
        self.logger.info(f"LLM final response: {final_answer}")

        # Logging retrieved documents
        retriever_output = response.get("retriever", {})
        retrieved_docs = retriever_output.get("documents", [])
        self.logger.info(f"Number of documents retrieved: {len(retrieved_docs)}")

        if retrieved_docs:
            # Log the details of the first retrieved document
            # The __str__ or __repr__ of the Haystack Document object will be logged.
            self.logger.info(f"Top retrieved document: {retrieved_docs[0]}")
            # You can also log specific fields if you prefer:
            # self.logger.info(f"Top retrieved document ID: {retrieved_docs[0].id}")
            # self.logger.info(f"Top retrieved document content snippet: {retrieved_docs[0].content[:200]}...")
            # self.logger.info(f"Top retrieved document score: {retrieved_docs[0].score}") # If score is available
        else:
            self.logger.info("No documents were retrieved.")

        # For more detailed logging of all retrieved documents (optional, can be verbose):
        # for i, doc in enumerate(retrieved_docs):
        #     self.logger.debug(f"Retrieved Doc {i+1} ID: {doc.id}, Score: {doc.score if hasattr(doc, 'score') else 'N/A'}")
        #     self.logger.debug(f"Retrieved Doc {i+1} Content: {doc.content[:100]}...")

        return final_answer
    
    