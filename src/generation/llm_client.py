"""
llm_client.py

Cliente que llama a la API de Claude con el prompt construido, y une
todas las piezas del pipeline (retrieval + generación) en una sola
función de conveniencia: answer_question().

Requiere la variable de entorno ANTHROPIC_API_KEY (ver .env.example).
"""

import os
from dataclasses import dataclass
from typing import List

from anthropic import Anthropic
from dotenv import load_dotenv

from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retrieval.retriever import Retriever, RetrievedChunk, format_chunks_for_prompt

load_dotenv()

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1024


@dataclass
class RAGAnswer:
    """Respuesta final del sistema, con la respuesta y las fuentes usadas."""
    answer: str
    sources: List[str]
    had_relevant_context: bool


class LLMClient:
    """Encapsula la llamada a la API de Claude para la fase de generación."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "No se encontró ANTHROPIC_API_KEY. Copia .env.example a .env "
                "y añade tu clave antes de continuar."
            )
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, query: str, formatted_context: str) -> str:
        """Llama al modelo con el system prompt y el contexto ya formateado."""
        user_prompt = build_user_prompt(query, formatted_context)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # el contenido puede incluir varios bloques; nos quedamos con los de texto
        text_blocks = [block.text for block in response.content if block.type == "text"]
        return "\n".join(text_blocks)


def answer_question(query: str, retriever: Retriever, llm_client: LLMClient) -> RAGAnswer:
    """
    Función de orquestación de extremo a extremo: recupera contexto,
    genera la respuesta, y devuelve todo junto con las fuentes citadas.

    Esta es la función que llamará directamente el endpoint de FastAPI.
    """
    retrieved_chunks: List[RetrievedChunk] = retriever.retrieve(query)

    if not retrieved_chunks:
        return RAGAnswer(
            answer="No tengo información suficiente en el corpus para responder a esto.",
            sources=[],
            had_relevant_context=False,
        )

    formatted_context = format_chunks_for_prompt(retrieved_chunks)
    answer_text = llm_client.generate(query, formatted_context)

    unique_sources = sorted({chunk.source for chunk in retrieved_chunks})

    return RAGAnswer(
        answer=answer_text,
        sources=unique_sources,
        had_relevant_context=True,
    )


if __name__ == "__main__":
    # Uso rápido de prueba: python -m src.generation.llm_client
    from src.embeddings.embedder import EmbeddingIndexer

    indexer = EmbeddingIndexer()
    retriever = Retriever(indexer)
    llm_client = LLMClient()

    result = answer_question("¿cuál es el plazo para presentar un reclamo?", retriever, llm_client)
    print(f"Respuesta: {result.answer}")
    print(f"Fuentes: {result.sources}")

