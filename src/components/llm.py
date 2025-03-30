from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

class LLMProvider(ABC):
    """Base class for LLM providers"""
    @abstractmethod
    def __init__(self, api_key: str, model_name: str, base_url: str = None, generation_kwargs: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize the LLM provider with api key and model name"""
        pass
    
    @abstractmethod
    def create_llm(self) -> Any:
        """Create and return a Haystack generator component"""
        pass

    @staticmethod
    def create_provider(provider_type: str, api_key: str, model_name: str, **kwargs) -> 'LLMProvider':
        """Factory method to create the appropriate provider"""
        if provider_type.lower() == "openai" or provider_type.lower() == "local":
            return LocalOpenAIProvider(api_key, model_name, **kwargs)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

class LocalOpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: str, model_name: str, base_url: str = None, generation_kwargs: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize the OpenAI provider"""
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.generation_kwargs = generation_kwargs or {}
        self.generation_kwargs.setdefault("temperature", 0.7)
        self.generation_kwargs.setdefault("max_tokens", 512)
        self.generator = None

    def create_llm(self) -> OpenAIChatGenerator:
        """Create and return an OpenAI generator component"""
        if self.generator is None:
            self.generator = OpenAIChatGenerator(
                model=self.model_name,
                api_key=Secret.from_token(self.api_key),
                api_base_url=self.base_url,
                generation_kwargs=self.generation_kwargs,
            )
        return self.generator