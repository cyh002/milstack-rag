from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

class BasePipeline(ABC):
    """Abstract base class for all pipelines in the system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.components = {}
        self.initialized = False
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize pipeline components."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the pipeline with given inputs."""
        pass
    
    def validate(self) -> bool:
        """Validate that pipeline is properly initialized."""
        if not self.initialized:
            self.logger.error("Pipeline not initialized. Call initialize() first.")
            return False
        return True