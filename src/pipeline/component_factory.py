from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import logging

from components.embedder import EmbedderFactory
from components.document_store import DocumentStoreFactory
from components.memory_store import MemoryStoreFactory
from components.llm import LLMProvider
from milvus_haystack import MilvusDocumentStore

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

    def __init__(self, config=None):
        """Initialize the factory with configuration."""
        self.config = config or {}
        self.vector_db_cfg = self.config.get("vector_db", {})
        self.llm_cfg = self.config.get("llm", {})
        self.embedding_cfg = self.config.get("embedding", {})
        
        # Initialize sub-factories
        self.embedder_factory = EmbedderFactory(self.embedding_cfg)
        self.document_store_factory = DocumentStoreFactory(self.vector_db_cfg)
        self.memory_store_factory = MemoryStoreFactory()

    def create_document_store(self) -> MilvusDocumentStore:
        return self.document_store_factory.create_document_store()
    
    def create_document_embedder(self, provider: str = "sentence_transformers", **kwargs) -> Any:
        return self.embedder_factory.create_document_embedder(provider, **kwargs)
    
    def create_text_embedder(self, provider: str = "sentence_transformers", **kwargs) -> Any:
        return self.embedder_factory.create_text_embedder(provider, **kwargs)
    
    def create_memory_components(self) -> Dict[str, Any]:
        return self.memory_store_factory.create_memory_components()
    
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
        return self.document_store_factory.create_retriever(document_store)
