import os
import logging
from omegaconf import OmegaConf
from pathlib import Path
from typing import Any, Dict, Optional, List, Set

class ConfigLoader:
    def __init__(self, config_path=None):
        
        # Load YAML config with OmegaConf if provided
        self.config = {}
        if config_path:
            # Load the config
            self.config = OmegaConf.load(config_path)
            
    def get_config(self):
        """Return the complete configuration"""
        return self.config
    
    def get(self, key, default=None):
        """Get a configuration value with dot notation support"""
        try:
            keys = key.split('.')
            value = self.config
            
            for k in keys:
                value = value[k]
                
            return value
        except (KeyError, TypeError, AttributeError):
            return default
    
    def get_llm_config(self):
        """Get the configuration for the currently selected LLM provider"""
        provider = os.environ.get('LLM_PROVIDER')
        return self.get(f"llm.{provider}", {})
    
    def log_environment_variables(self, mask_secrets=True, prefix_filter=None):
        """Log all environment variables, optionally masking sensitive ones
        
        Args:
            mask_secrets: If True, mask API keys and passwords
            prefix_filter: Optional list of prefixes to filter by (e.g. ['OPENAI_', 'VLLM_'])
        """
        # Define sensitive key patterns
        sensitive_patterns = ['api_key', 'apikey', 'password', 'secret', 'token']
        
        # Header for log output
        logging.info("===== Environment Variables =====")
        
        # Get all environment variables
        for key, value in sorted(os.environ.items()):
            # Skip if doesn't match prefix filter
            if prefix_filter and not any(key.startswith(p) for p in prefix_filter):
                continue
                
            # Mask sensitive values
            if mask_secrets and any(pattern in key.lower() for pattern in sensitive_patterns):
                # Show first 4 and last 4 chars if long enough
                if len(value) > 10:
                    masked_value = f"{value[:4]}...{value[-4:]}"
                else:
                    masked_value = "********"
                logging.info(f"  {key} = {masked_value}")
            else:
                logging.info(f"  {key} = {value}")
                
        logging.info("================================")