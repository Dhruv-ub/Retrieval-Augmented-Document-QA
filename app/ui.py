"""
Application UI Module.
FAANG Pattern: Clean separation of UI from business logic.
Uses Gradio 5.x 'messages' format.
"""

import gradio as gr
import time
from typing import Callable, List, Dict
from dataclasses import dataclass


@dataclass
class UIConfig:
    """UI configuration."""
    title: str = "FAANG-Level RAG System"
    description: str = """
**Production-Grade Retrieval-Augmented Generation**

Architecture: Modular Python Packages | Model: TinyLlama-1.1B | Vector Store: FAISS

Features:
- Semantic document retrieval
- Grounded answer generation
- Hallucination detection
- Performance metrics
"""
    theme: str = "soft"


def create_ui(pipeline_func: Callable, config: UIConfig = None) -> gr.Blocks:
    """
    Create Gradio UI for RAG system.

    Args:
        pipeline_func: RAG pipeline function (query -> response dict)
        config: UI configuration

    Returns:
        Gradio Blocks interface
    """
    config = config or UIConfig()

    def process_query(message: str, history: List[Dict]) -> List[Dict]:
        """
        Process user query and return updated chat history.
        Uses list of dictionaries for Gradio 5.x compatibility.
        """
        if not message.strip():
            return history

        # Append User Message
        history.append({"role": "user", "content": message})

        try:
            start_time = time.perf_counter()
            response = pipeline_func(message)
            latency = time.perf_counter() - start_time

            # Safely get response data
            answer = response.get("answer", "No answer generated.")
            grounding = response.get("grounding_score", 0.0)
            sources = response.get("source_pages", [])
            retrieval_time = response.get("retrieval_time_ms", 0)

            # Grounding Score Indicators
            if grounding >= 0.7:
                g_emoji, g_label = "G-High", "High"
            elif grounding >= 0.4:
                g_emoji, g_label = "G-Med", "Medium"
            else:
                g_emoji, g_label = "G-Low", "Low"

            # Formatted Output
            output = f"""{answer}

---
**System Metrics**
| Metric | Value |
|--------|-------|
| Total Latency | {latency:.2f}s |
| Grounding Score | {g_emoji} {grounding:.2f} ({g_label}) |
| Source Pages | {sources} |
"""
            # Append Assistant Response
            history.append({"role": "assistant", "content": output})
            return history

        except Exception as e:
            history.append({"role": "assistant", "content": f"Error: {str(e)}"})
            return history

    # Build the UI
    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown(f"# {config.title}")
        gr.Markdown(config.description)

        # type="messages" is required for Gradio 5.x
        chatbot = gr.Chatbot(type="messages", height=600)

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask a question about the document...",
                scale=4,
                show_label=False
            )
            clear_btn = gr.Button("Clear", scale=1)

        # Handle Submission
        msg.submit(
            process_query,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            lambda: "", None, msg  # Clear input box
        )

        clear_btn.click(lambda: [], None, chatbot)

    return interface
