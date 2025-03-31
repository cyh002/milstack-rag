from typing import Dict, Any

# Memory components
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
from haystack import component
from typing import Any
from haystack.core.component.types import Variadic
from itertools import chain

@component
class ListJoiner:
    """Joins multiple lists into a single list.
    This is used to handle messages from both the user and LLM, writing them to the memory store.
    """
    def __init__(self, _type: Any):
        component.set_output_types(self, values=_type)

    def run(self, values: Variadic[Any]):
        result = list(chain(*values))
        return {"values": result}

class MemoryStoreFactory:
    """Factory for creating memory store components."""
    
    def create_memory_components(self) -> Dict[str, Any]:
        """Creates and returns memory components."""
        memory_store = InMemoryChatMessageStore()
        return {
            "memory_store": memory_store,
            "memory_retriever": ChatMessageRetriever(memory_store),
            "memory_writer": ChatMessageWriter(memory_store)
        }