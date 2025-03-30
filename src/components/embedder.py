from typing import Any, Dict
import logging

# Haystack embedders
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.embedders import OpenAITextEmbedder, OpenAIDocumentEmbedder
from haystack.utils import Secret

class EmbedderFactory:
    """Factory for creating embedder components."""
    
    def __init__(self, embedding_cfg: Dict[str, Any] = None):
        self.embedding_cfg = embedding_cfg or {}
    
    def create_document_embedder(self, provider: str = "sentence_transformers", **kwargs) -> Any:
        """Creates a document embedder based on provider.
        
        Args:
            provider: One of 'sentence_transformers', 'openai'
            **kwargs: Additional parameters for the embedder
        """
        logging.info(f"Creating document embedder with provider: {provider}")
        sentence_transformers_cfg = self.embedding_cfg.get("sentence_transformers", {})
        
        if provider == "sentence_transformers":
            model_name = kwargs.get("document_embedder", sentence_transformers_cfg.get("document_embedder"))
            logging.info(f"Creating SentenceTransformersDocumentEmbedder with model: {model_name}")
            return SentenceTransformersDocumentEmbedder(model=model_name)
        
        elif provider == "openai":
            openai_embedding_cfg = self.embedding_cfg.get("openai", {})
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
            provider: One of 'sentence_transformers', 'openai'
            **kwargs: Additional parameters for the embedder
        """
        if provider == "sentence_transformers":
            sentence_transformers_cfg = self.embedding_cfg.get("sentence_transformers", {})
            model_name = kwargs.get("text_embedder", sentence_transformers_cfg.get("text_embedder"))
            logging.info(f"Creating SentenceTransformersTextEmbedder with model: {model_name}")
            return SentenceTransformersTextEmbedder(model=model_name)
        
        elif provider == "openai":
            openai_embedding_cfg = self.embedding_cfg.get("openai", {})
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