"""
Utility functions for C4Engine.
"""

import os
from typing import Optional
from dotenv import load_dotenv


def load_config() -> dict:
    """
    Load configuration from environment variables.
    
    Returns:
        Dictionary of configuration values
    """
    load_dotenv()
    
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "model": os.getenv("MODEL", "gpt-4"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "2000")),
        "max_iterations": int(os.getenv("MAX_ITERATIONS", "5")),
        "temperature": float(os.getenv("TEMPERATURE", "0.7")),
    }
    
    return config


def setup_api_keys(openai_key: Optional[str] = None, anthropic_key: Optional[str] = None):
    """
    Set up API keys for LLM providers.
    
    Args:
        openai_key: OpenAI API key
        anthropic_key: Anthropic API key
    """
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key


def validate_config(config: dict) -> bool:
    """
    Validate configuration values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ["openai_api_key"]
    
    for key in required_keys:
        if not config.get(key):
            print(f"Warning: Missing required configuration: {key}")
            return False
    
    return True
