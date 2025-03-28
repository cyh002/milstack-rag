from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from haystack.dataclasses import ChatMessage

class TemplateProvider(ABC):
    """Abstract base class for providing templates."""
    
    @abstractmethod
    def get_system_message(self) -> ChatMessage:
        """Returns the system message."""
        pass
    
    @abstractmethod
    def get_main_template(self) -> List[ChatMessage]:
        """Returns the main RAG template."""
        pass
    
    @abstractmethod
    def get_query_rephrase_template(self) -> str:
        """Returns the query rephrase template."""
        pass

class SevenWondersTemplateProvider(TemplateProvider):
    """Template provider for Seven Wonders dataset."""
    
    def get_system_message(self) -> ChatMessage:
        return ChatMessage.from_system(
            "You are a helpful AI assistant using provided supporting documents and conversation history to assist humans"
        )
    
    def get_main_template(self) -> List[ChatMessage]:
        return [
            self.get_system_message(),
            ChatMessage.from_user(
                """Given the conversation history and the provided supporting documents, give a brief answer to the question.
                Note that supporting documents are not part of the conversation. If question can't be answered from supporting documents, say so.

                Conversation history:
                {% for memory in memories %}
                    {{ memory.text }}
                {% endfor %}

                Supporting documents:
                {% for doc in documents %}
                    {{ doc.content }}
                {% endfor %}

                Question: {{ question }}
                Answer:"""
            )
        ]
    
    def get_query_rephrase_template(self) -> str:
        return """
            Rewrite the question for search while keeping its meaning and key terms intact.
            If the conversation history is empty, DO NOT change the query.
            Use conversation history only if necessary, and avoid extending the query with your own knowledge.
            If no changes are needed, output the current question as is.

            Conversation history:
            {% for memory in memories %}
                {{ memory.text }}
            {% endfor %}

            User Query: {{query}}
            Rewritten Query:
        """
