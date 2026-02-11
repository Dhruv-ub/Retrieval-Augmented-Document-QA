"""
LLM Engine Module.
FAANG Pattern: Facade Pattern - simplifies complex LLM interaction.
Uses Float16 mode for stable inference.
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1


@dataclass
class GenerationResult:
    """Result container for LLM generation."""
    text: str
    tokens_generated: int
    generation_time_ms: float
    model_name: str


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""

    @abstractmethod
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Generate text from prompt."""
        pass


class TinyLlamaEngine(BaseLLM):
    """
    Production LLM Engine using TinyLlama.

    MODE: Float16 (Standard)
    - Uses float16 for stable inference without quantization dependencies.
    - Fits easily on T4 GPU (Model size ~2.2GB, GPU RAM 15GB).
    """

    def __init__(
        self,
        model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map: str = "auto"
    ):
        self.model_id = model_id
        self.device_map = device_map
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load model and tokenizer."""
        if self._model is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            print(f"Loading {self.model_id} in Float16 mode...")

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            # Load directly in Float16 (stable, no quantization dependencies)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=self.device_map,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        max_new_tokens: int = 256
    ) -> str:
        import torch

        self._load_model()
        config = config or GenerationConfig(max_new_tokens=max_new_tokens)

        # Tokenize input
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.do_sample,
                repetition_penalty=config.repetition_penalty,
                pad_token_id=self._tokenizer.eos_token_id
            )

        # Decode output
        full_output = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "[/INST]" in full_output:
            return full_output.split("[/INST]")[-1].strip()
        return full_output[len(prompt):].strip()


# Backward compatibility alias
class LLMEngine(TinyLlamaEngine):
    pass
