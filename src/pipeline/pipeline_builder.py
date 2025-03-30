from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

import logging
from haystack import Pipeline
from haystack.dataclasses import ChatMessage

from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import OutputAdapter

from components.list_joiner import ListJoiner

from pipeline.template_provider import TemplateProvider
from pipeline.component_factory import ComponentFactory

from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever

class PipelineBuilder:
    """Builder for constructing RAG pipelines."""
    
    def __init__(self, component_factory: ComponentFactory, template_provider: TemplateProvider):
        self.component_factory = component_factory
        self.template_provider = template_provider
        self.pipeline = Pipeline()
        self.components = {}
    
    def add_components(self, model_name: str, api_base_url: str, 
                      temperature: float = 0.7, max_tokens: int = 512) -> 'PipelineBuilder':
        """Add all components to the pipeline."""
        # Memory components
        memory_components = self.component_factory.create_memory_components()
        self.components.update(memory_components)
        
        # Text embedder
        text_embedder = self.component_factory.create_text_embedder()
        self.components["text_embedder"] = text_embedder
        
        # Query rephrasing components
        self.components["query_rephrase_prompt_builder"] = ChatPromptBuilder(
            template=self.template_provider.get_query_rephrase_template())
        
        self.components["query_rephrase_llm"] = self.component_factory.create_llm(
            model_name, api_base_url, temperature, max_tokens)
        
        self.components["list_to_str_adapter"] = OutputAdapter(
            template="{{ replies[0] }}", output_type=str)
        
        # RAG components - use the document_store as a parameter, not a component
        document_store = self.components.get("document_store")
        self.components["retriever"] = self.component_factory.create_retriever(document_store)
        
        self.components["prompt_builder"] = ChatPromptBuilder(
            template=self.template_provider.get_main_template(),
            required_variables=["question", "documents", "memories"])
        
        # LLM component
        self.components["llm"] = self.component_factory.create_llm(
            model_name, api_base_url, temperature, max_tokens)
        
        # Memory joiner
        self.components["memory_joiner"] = ListJoiner(List[ChatMessage])
        
        # Add only actual components to pipeline
        component_names = [
            "memory_retriever", "memory_writer", "text_embedder", 
            "query_rephrase_prompt_builder", "query_rephrase_llm", "list_to_str_adapter",
            "retriever", "prompt_builder", "llm", "memory_joiner"
        ]
        
        for name in component_names:
            if name in self.components:
                self.pipeline.add_component(name, self.components[name])
        
        return self
    
    def connect_components(self) -> 'PipelineBuilder':
        """Connect all components in the pipeline."""
        # Query rephrasing connections
        self.pipeline.connect("memory_retriever", "query_rephrase_prompt_builder.memories")
        self.pipeline.connect("query_rephrase_prompt_builder.prompt", "query_rephrase_llm")
        self.pipeline.connect("query_rephrase_llm.replies", "list_to_str_adapter")
        
        # Connect embedding to retriever
        self.pipeline.connect("list_to_str_adapter", "text_embedder.text")
        self.pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

        # RAG connections
        self.pipeline.connect("retriever.documents", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder.prompt", "llm.messages")
        self.pipeline.connect("llm.replies", "memory_joiner")

        # Memory connections
        self.pipeline.connect("memory_joiner", "memory_writer")
        self.pipeline.connect("memory_retriever", "prompt_builder.memories")
        
        return self
    
    def build(self) -> Pipeline:
        """Return the built pipeline."""
        return self.pipeline
