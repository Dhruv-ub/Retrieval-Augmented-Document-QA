"""
Prompt Engineering Module.
FAANG Pattern: Template Method Pattern for consistent prompts.
"""
from typing import List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class PromptContext:
    """Container for prompt context information."""
    context: str
    question: str
    system_instruction: Optional[str] = None
    examples: Optional[List[dict]] = None


class BasePromptTemplate(ABC):
    """Abstract base class for prompt templates."""

    @abstractmethod
    def format(self, context: PromptContext) -> str:
        """Format the prompt template."""
        pass


class RAGPromptTemplate(BasePromptTemplate):
    """
    Production RAG prompt template.

    FAANG Patterns:
    - Clear instruction hierarchy
    - Explicit grounding requirements
    - Structured output guidance
    """

    SYSTEM_PROMPT = """You are a precise technical assistant. Your task is to answer questions
based ONLY on the provided context. Follow these rules strictly:

1. ONLY use information from the context below
2. If the answer is not in the context, say "I cannot find this information in the provided documents"
3. Quote relevant parts of the context when possible
4. Be concise but complete
5. If uncertain, express your uncertainty"""

    def format(self, context: PromptContext) -> str:
        """
        Format the RAG prompt.

        Args:
            context: PromptContext containing question and context

        Returns:
            Formatted prompt string
        """
        system_instruction = context.system_instruction or self.SYSTEM_PROMPT

        prompt = f"""[INST] {system_instruction}

Context:
---
{context.context}
---

Question: {context.question}

Answer: [/INST]"""

        return prompt

    @staticmethod
    def get_rag_prompt(context: str, question: str) -> str:
        """Convenience method for simple RAG prompts."""
        template = RAGPromptTemplate()
        return template.format(PromptContext(context=context, question=question))


# Backward compatibility alias
class PromptTemplate:
    @staticmethod
    def get_rag_prompt(context: str, question: str) -> str:
        return RAGPromptTemplate.get_rag_prompt(context, question)
