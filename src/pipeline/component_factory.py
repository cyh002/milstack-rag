from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder

# Conversational Memory: 
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter 


class ComponentFactory(ABC):
    """Abstract factory for creating pipeline components."""
    
    @abstractmethod
    def create_document_store(self) -> InMemoryDocumentStore:
        """Creates and returns a document store."""
        pass
    
    @abstractmethod
    def create_document_embedder(self) -> SentenceTransformersDocumentEmbedder:
        """Creates and returns a document embedder."""
        pass
    
    @abstractmethod
    def create_text_embedder(self) -> SentenceTransformersTextEmbedder:
        """Creates and returns a text embedder."""
        pass
    
    @abstractmethod
    def create_memory_components(self) -> Dict[str, Any]:
        """Creates and returns memory components."""
        pass
    
    @abstractmethod
    def create_llm(self, model_name: str, api_base_url: str, **kwargs) -> OpenAIChatGenerator:
        """Creates and returns an LLM."""
        pass


class RAGComponentFactory(ComponentFactory):
    """Factory for creating RAG pipeline components."""
    
    def create_document_store(self) -> InMemoryDocumentStore:
        return InMemoryDocumentStore(embedding_similarity_function="cosine")
    
    def create_document_embedder(self) -> SentenceTransformersDocumentEmbedder:
        return SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def create_text_embedder(self) -> SentenceTransformersTextEmbedder:
        return SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def create_memory_components(self) -> Dict[str, Any]:
        memory_store = InMemoryChatMessageStore()
        return {
            "memory_store": memory_store,
            "memory_retriever": ChatMessageRetriever(memory_store),
            "memory_writer": ChatMessageWriter(memory_store)
        }
    
    def create_llm(self, model_name: str, api_base_url: str, 
                 temperature: float = 0.7, max_tokens: int = 512) -> OpenAIChatGenerator:
        return OpenAIChatGenerator(
            api_key=Secret.from_token("token-abc123"),
            model=model_name,
            api_base_url=api_base_url,
            generation_kwargs={
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
