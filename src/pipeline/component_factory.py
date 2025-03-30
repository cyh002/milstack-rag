from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import os
from pathlib import Path

# Haystack embedders
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.embedders import OpenAITextEmbedder, OpenAIDocumentEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret

# Milvus
from milvus_haystack import MilvusDocumentStore
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever

# Memory components
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
from components.llm import LLMProvider
import logging

class ComponentFactory(ABC):
    """Abstract factory for creating pipeline components."""
    
    @abstractmethod
    def create_document_store(self) -> MilvusDocumentStore:
        """Creates and returns a document store."""
        pass
    
    @abstractmethod
    def create_document_embedder(self, provider: str = "sentence_transformers") -> Any:
        """Creates and returns a document embedder."""
        pass
    
    @abstractmethod
    def create_text_embedder(self, provider: str = "sentence_transformers") -> Any:
        """Creates and returns a text embedder."""
        pass
    
    @abstractmethod
    def create_memory_components(self) -> Dict[str, Any]:
        """Creates and returns memory components."""
        pass
    
    @abstractmethod
    def create_llm(self, model_name: str, api_base_url: str, **kwargs) -> Any:
        """Creates and returns an LLM."""
        pass

class RAGComponentFactory(ComponentFactory):
    """Factory for creating RAG pipeline components."""

    def __init__(self, 
                 config=None):
        """Initialize the factory with configuration."""
        self.config = config
        self.vector_db_cfg = self.config.get("vector_db", {})
        self.llm_cfg = self.config.get("llm", {})
        self.embedding_cfg = self.config.get("embedding", {})

    def create_document_store(self) -> MilvusDocumentStore:
        """Creates a Milvus document store.
        
        Args:
            db_path: Path to Milvus database
            collection_name: Name of the collection
            recreate: If True, drop and recreate the collection
        """
        if self.vector_db_cfg.get("load_type") == 'milvus-lite':
            db_path = self.vector_db_cfg.get("db_path")
            if not db_path:
                raise ValueError("Database path is not set. Please provide a valid path.")
            # Continue with db_path setup
        collection_name = self.vector_db_cfg.get("collection_name")
        recreate = self.vector_db_cfg.get("recreate")
        dimension = self.vector_db_cfg.get("dimension")
        logging.info(f"Creating Milvus document store with db_path: {db_path}, "
                     f"collection_name: {collection_name}, recreate: {recreate}, "
                     f"dimension: {dimension}")
        # Create the document store
        return MilvusDocumentStore(
            connection_args={"uri": db_path},
            collection_name=collection_name,
            drop_old=recreate
        )
    
    def create_document_embedder(self, provider: str = "sentence_transformers", **kwargs) -> Any:
        """Creates a document embedder based on provider.
        
        Args:
            provider: One of 'sentence_transformers', 'openai', 'gemini'
            **kwargs: Additional parameters for the embedder
        """
        logging.info(f"Creating document embedder with provider: {provider}")
        sentence_transformers_cfg = self.embedding_cfg.get("sentence_transformers")
        if provider == "sentence_transformers":
            model_name = kwargs.get("document_embedder", sentence_transformers_cfg.get("document_embedder"))
            logging.info(f"Creating SentenceTransformersDocumentEmbedder with model: {model_name}")
            return SentenceTransformersDocumentEmbedder(model=model_name)
        
        elif provider == "openai":
            openai_embedding_cfg = self.embedding_cfg.get("openai")
            api_key = kwargs.get("api_key", openai_embedding_cfg.get("api_key"))
            if not api_key:
                raise ValueError("Document Embedding API key is not set. Please provide a valid OpenAI API key.")
            model = kwargs.get("model", openai_embedding_cfg.get("model_name"))
            if not model:
                raise ValueError("Document Embedding model name is not set. Please provide a valid OpenAI model name.")
            logging.info(f"Creating OpenAIDocumentEmbedder with model: {model}")
            return OpenAIDocumentEmbedder(api_key=Secret.from_token(api_key), model=model)
        
        else:
            raise ValueError(f"Unsupported embedder provider: {provider}")
    
    def create_text_embedder(self, provider: str = "sentence_transformers", **kwargs) -> Any:
        """Creates a text embedder based on provider.
        
        Args:
            provider: One of 'sentence_transformers', 'openai', 'gemini'
            **kwargs: Additional parameters for the embedder
        """
        if provider == "sentence_transformers":
            sentence_transformers_cfg = self.embedding_cfg.get("sentence_transformers")
            model_name = kwargs.get("text_embedder", sentence_transformers_cfg.get("text_embedder"))
            logging.info(f"Creating SentenceTransformersTextEmbedder with model: {model_name}")
            return SentenceTransformersTextEmbedder(model=model_name)
        
        elif provider == "openai":
            openai_embedding_cfg = self.embedding_cfg.get("openai")
            api_key = kwargs.get("api_key", openai_embedding_cfg.get("api_key"))
            if not api_key:
                raise ValueError("Text Embedding API key is not set. Please provide a valid OpenAI API key.")
            model = kwargs.get("model", openai_embedding_cfg.get("model_name"))
            if not model:
                raise ValueError("Text Embedding model name is not set. Please provide a valid OpenAI model name.")
            logging.info(f"Creating OpenAITextEmbedder with model: {model}")
            return OpenAITextEmbedder(api_key=Secret.from_token(api_key), model=model)
        
        else:
            raise ValueError(f"Unsupported embedder provider: {provider}")
    
    def create_memory_components(self) -> Dict[str, Any]:
        memory_store = InMemoryChatMessageStore()
        return {
            "memory_store": memory_store,
            "memory_retriever": ChatMessageRetriever(memory_store),
            "memory_writer": ChatMessageWriter(memory_store)
        }
    
    def create_llm(self, model_name: str = None, api_base_url: str = None, 
                 temperature: float = 0.7, max_tokens: int = 512) -> Any:
        """Creates and returns an LLM using the LLMProvider.
        
        Args:
            model_name: Optional override for model name
            api_base_url: Optional override for API base URL
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
        
        Returns:
            A Haystack generator component
        """
        # Get the provider type from config
        provider_type = self.llm_cfg.get("provider", "local")
        # Get provider-specific config
        provider_config = self.llm_cfg.get(provider_type, {})
    
        # Use passed parameters or get from config
        model_name = model_name or provider_config.get("model_name")
        
        # Handle different naming conventions in config (base_url vs api_base)
        if api_base_url is None:
            api_base_url = provider_config.get("base_url")
            if api_base_url is None and "api_base" in provider_config:
                api_base_url = provider_config.get("api_base")
        
        api_key = provider_config.get("api_key", "token-abc123")
        
        # Get generation parameters from pipeline config or use defaults
        pipeline_config = self.config.get("pipeline", {}).get("generator", {})
        temperature = pipeline_config.get("temperature", temperature)
        max_tokens = pipeline_config.get("max_tokens", max_tokens)
        
        # Create generation kwargs
        generation_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Create the LLM provider and return the LLM
        llm_provider = LLMProvider.create_provider(
            provider_type=provider_type,
            api_key=api_key,
            model_name=model_name,
            base_url=api_base_url,
            generation_kwargs=generation_kwargs
        )
        return llm_provider.create_llm()
    
    def create_retriever(self, document_store):
        """Creates and returns a retriever for the document store."""
        return MilvusEmbeddingRetriever(document_store=document_store, top_k=5)
