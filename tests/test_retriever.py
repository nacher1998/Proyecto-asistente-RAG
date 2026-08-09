"""
Tests para el módulo de retrieval.

Usamos un EmbeddingIndexer real (con Chroma) pero controlamos los
resultados directamente vía un stub, para no depender de descargar un
modelo de embeddings real durante los tests (los tests deben ser
rápidos y no requerir red).
"""

from unittest.mock import MagicMock

from src.retrieval.retriever import Retriever, format_chunks_for_prompt


def make_fake_query_result(documents, metadatas, distances):
    """Construye un resultado con la misma forma que devuelve Chroma."""
    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }


def test_retrieve_filters_by_relevance_threshold():
    fake_indexer = MagicMock()
    fake_indexer.query.return_value = make_fake_query_result(
        documents=["texto relevante", "texto irrelevante"],
        metadatas=[{"source": "a.txt", "chunk_index": 0}, {"source": "b.txt", "chunk_index": 0}],
        distances=[0.3, 0.9],  # el segundo supera el umbral por defecto (0.7)
    )

    retriever = Retriever(indexer=fake_indexer, relevance_threshold=0.7)
    results = retriever.retrieve("pregunta de prueba")

    assert len(results) == 1
    assert results[0].source == "a.txt"


def test_retrieve_returns_empty_when_nothing_relevant():
    fake_indexer = MagicMock()
    fake_indexer.query.return_value = make_fake_query_result(
        documents=["texto irrelevante"],
        metadatas=[{"source": "a.txt", "chunk_index": 0}],
        distances=[0.95],
    )

    retriever = Retriever(indexer=fake_indexer, relevance_threshold=0.7)
    results = retriever.retrieve("pregunta fuera de dominio")

    assert results == []


def test_has_relevant_context():
    fake_indexer = MagicMock()
    fake_indexer.query.return_value = make_fake_query_result(
        documents=["texto relevante"],
        metadatas=[{"source": "a.txt", "chunk_index": 0}],
        distances=[0.2],
    )

    retriever = Retriever(indexer=fake_indexer)
    assert retriever.has_relevant_context("pregunta") is True


def test_format_chunks_for_prompt_includes_source():
    fake_indexer = MagicMock()
    retriever = Retriever(indexer=fake_indexer)
    chunks = retriever.retrieve  # no se usa directamente aquí

    from src.retrieval.retriever import RetrievedChunk
    sample = [RetrievedChunk(text="contenido X", source="doc.txt", chunk_index=0, distance=0.1)]

    formatted = format_chunks_for_prompt(sample)
    assert "doc.txt" in formatted
    assert "contenido X" in formatted


def test_format_chunks_for_prompt_handles_empty_list():
    formatted = format_chunks_for_prompt([])
    assert "No se encontró información relevante" in formatted
