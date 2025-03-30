from typing import Dict, Any

# Memory components
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter

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