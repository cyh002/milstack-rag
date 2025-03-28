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
