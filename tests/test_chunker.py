"""
Tests básicos para el módulo de chunking.

Estos tests no requieren archivos reales en disco: construimos
RawDocument manualmente para probar la lógica de troceado de forma
aislada y rápida.
"""

import pytest

from src.ingestion.loader import RawDocument
from src.ingestion.chunker import chunk_document, chunk_documents


def make_document(num_words: int, source: str = "test.txt") -> RawDocument:
    """Crea un RawDocument de prueba con un número exacto de palabras."""
    text = " ".join(f"palabra{i}" for i in range(num_words))
    return RawDocument(source=source, text=text, doc_type="txt")


def test_chunk_document_produces_expected_count():
    doc = make_document(num_words=1000)
    chunks = chunk_document(doc, chunk_size=500, overlap=50)

    # con 1000 palabras, chunk_size=500 y overlap=50, el paso efectivo es 450
    # así que esperamos más de un chunk
    assert len(chunks) >= 2
    assert all(chunk.source == "test.txt" for chunk in chunks)


def test_chunks_have_overlap():
    doc = make_document(num_words=1000)
    chunks = chunk_document(doc, chunk_size=500, overlap=50)

    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()

    # las últimas 50 palabras del primer chunk deben coincidir con
    # las primeras 50 del segundo chunk
    assert first_words[-50:] == second_words[:50]


def test_empty_document_returns_no_chunks():
    doc = RawDocument(source="vacio.txt", text="", doc_type="txt")
    chunks = chunk_document(doc)
    assert chunks == []


def test_overlap_must_be_smaller_than_chunk_size():
    doc = make_document(num_words=100)
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=100, overlap=100)


def test_chunk_documents_aggregates_multiple_docs():
    docs = [make_document(600, source="a.txt"), make_document(600, source="b.txt")]
    chunks = chunk_documents(docs, chunk_size=500, overlap=50)

    sources = {c.source for c in chunks}
    assert sources == {"a.txt", "b.txt"}

