"""
retriever.py

Dada una pregunta del usuario, busca en Chroma los fragmentos más
relevantes y decide cuáles son lo bastante buenos como para pasárselos
al LLM.

Por qué existe un umbral de relevancia (relevance_threshold):
Chroma siempre devuelve los top_k resultados más cercanos, incluso si
NINGUNO es realmente relevante para la pregunta (por ejemplo, si el
usuario pregunta algo totalmente fuera del dominio del corpus). Sin un
filtro, el LLM recibiría fragmentos irrelevantes como si fueran
contexto válido, lo que aumenta el riesgo de alucinación. Por eso
filtramos por distancia antes de pasar nada a generación.

Nota sobre la métrica: Chroma con el backend por defecto (HNSW) usa
distancia, no similitud — valores MÁS BAJOS significan MÁS parecido.
Esto es una fuente común de confusión, así que se documenta aquí
explícitamente.
"""

from dataclasses import dataclass
from typing import List

from src.embeddings.embedder import EmbeddingIndexer


# Umbral de distancia por defecto. Con embeddings normalizados y distancia
# coseno, valores por debajo de ~0.5 suelen indicar buena relevancia, pero
# ESTE VALOR SE DEBE CALIBRAR con tu propio corpus y preguntas de prueba
# (ver evaluation/eval_questions.json en el proyecto completo).
DEFAULT_RELEVANCE_THRESHOLD = 0.7
DEFAULT_TOP_K = 4


@dataclass
class RetrievedChunk:
    """Un fragmento recuperado, con su fuente y qué tan relevante fue."""
    text: str
    source: str
    chunk_index: int
    distance: float  # menor = más relevante


class Retriever:
    """Envuelve el EmbeddingIndexer para exponer una API de retrieval limpia."""

    def __init__(
        self,
        indexer: EmbeddingIndexer,
        top_k: int = DEFAULT_TOP_K,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ):
        self.indexer = indexer
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedChunk]:
        """
        Busca los fragmentos más relevantes para una pregunta.

        Devuelve una lista vacía si ningún fragmento supera el umbral
        de relevancia — esto es una señal para que la capa de generación
        responda "no tengo información suficiente" en vez de inventar.
        """
        k = top_k or self.top_k
        raw_results = self.indexer.query(query, top_k=k)

        chunks: List[RetrievedChunk] = []
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        for doc_text, metadata, distance in zip(documents, metadatas, distances):
            if distance <= self.relevance_threshold:
                chunks.append(
                    RetrievedChunk(
                        text=doc_text,
                        source=metadata.get("source", "desconocido"),
                        chunk_index=metadata.get("chunk_index", -1),
                        distance=distance,
                    )
                )

        return chunks

    def has_relevant_context(self, query: str, top_k: int = None) -> bool:
        """Atajo para saber si vale la pena seguir con la generación."""
        return len(self.retrieve(query, top_k=top_k)) > 0


def format_chunks_for_prompt(chunks: List[RetrievedChunk]) -> str:
    """
    Formatea los chunks recuperados como texto plano para insertarlos en
    el prompt del LLM, numerados y con su fuente, para que el modelo
    pueda citarlos.
    """
    if not chunks:
        return "No se encontró información relevante en el corpus."

    formatted_parts = []
    for i, chunk in enumerate(chunks, start=1):
        formatted_parts.append(
            f"[Fragmento {i} — fuente: {chunk.source}]\n{chunk.text}"
        )
    return "\n\n".join(formatted_parts)


if __name__ == "__main__":
    # Uso rápido de prueba: python -m src.retrieval.retriever
    from src.embeddings.embedder import EmbeddingIndexer

    indexer = EmbeddingIndexer()
    retriever = Retriever(indexer)

    resultados = retriever.retrieve("¿cuál es el plazo para presentar un reclamo?")
    print(format_chunks_for_prompt(resultados))

