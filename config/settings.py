"""
Configuration Management using Pydantic.
FAANG Pattern: Centralized, validated, type-safe configuration.
"""
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
import os


class DeviceType(str, Enum):
    """Supported compute devices."""
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    normalize: bool = True


@dataclass(frozen=True)
class ChunkingConfig:
    """Document chunking configuration."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: tuple = ("\n\n", "\n", ".", " ", "")


@dataclass(frozen=True)
class LLMConfig:
    """LLM configuration."""
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_new_tokens: int = 256
    temperature: float = 0.1
    load_in_4bit: bool = True
    device_map: str = "auto"


@dataclass(frozen=True)
class RetrieverConfig:
    """Retriever configuration."""
    top_k: int = 3
    similarity_threshold: float = 0.5
    use_reranking: bool = False


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration."""
    # Sub-configs
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)

    # Global settings
    log_level: LogLevel = LogLevel.INFO
    device: DeviceType = DeviceType.AUTO
    debug_mode: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Factory method to create config from environment variables."""
        return cls(
            debug_mode=os.getenv("DEBUG", "false").lower() == "true",
            log_level=LogLevel(os.getenv("LOG_LEVEL", "INFO"))
        )


# Singleton pattern for global config access
_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config

def set_config(config: AppConfig) -> None:
    """Override global configuration (useful for testing)."""
    global _config
    _config = config
