"""
chunker.py

Divide los documentos cargados en fragmentos (chunks) manejables para
generar embeddings y hacer retrieval.

Por qué trocear con solapamiento:
- Los LLMs y modelos de embeddings tienen un límite de contexto, así que
  no podemos pasar el documento completo.
- Trocear SIN solapamiento puede cortar una idea justo a la mitad entre
  dos fragmentos, perdiendo contexto. El solapamiento (overlap) mitiga
  esto: los últimos N caracteres de un chunk se repiten al inicio del
  siguiente.

Esta implementación trocea por palabras (más simple y predecible que
por tokens exactos de un tokenizer específico), lo cual es suficiente
para un proyecto de portfolio. Si quisieras ser más preciso con el
límite de tokens del modelo, se podría cambiar a un tokenizer real
(p. ej. tiktoken) sin tocar el resto del pipeline.
"""

from dataclasses import dataclass, field
from typing import List

from src.ingestion.loader import RawDocument


@dataclass
class Chunk:
    """Un fragmento de texto listo para generar su embedding."""
    id: str               # identificador único, p. ej. "documento.pdf_chunk_3"
    text: str
    source: str           # de qué documento proviene (para citar la fuente)
    chunk_index: int       # posición del chunk dentro del documento
    metadata: dict = field(default_factory=dict)


def split_into_words(text: str) -> List[str]:
    """Normaliza espacios y separa el texto en palabras."""
    return text.split()


def chunk_document(
    document: RawDocument,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Trocea un único documento en fragmentos de tamaño ~chunk_size palabras,
    con `overlap` palabras compartidas entre chunks consecutivos.

    Args:
        document: el RawDocument a trocear.
        chunk_size: número aproximado de palabras por fragmento.
        overlap: número de palabras que se repiten entre chunks consecutivos.

    Returns:
        Lista de Chunk generados a partir de este documento.
    """
    if overlap >= chunk_size:
        raise ValueError("El solapamiento debe ser menor que el tamaño del chunk.")

    words = split_into_words(document.text)
    if not words:
        return []

    chunks: List[Chunk] = []
    start = 0
    chunk_index = 0
    step = chunk_size - overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append(
            Chunk(
                id=f"{document.source}_chunk_{chunk_index}",
                text=chunk_text,
                source=document.source,
                chunk_index=chunk_index,
                metadata={**document.metadata, "doc_type": document.doc_type},
            )
        )

        chunk_index += 1
        start += step

        # evita bucle infinito si step <= 0 por configuración incorrecta
        if step <= 0:
            break

    return chunks


def chunk_documents(
    documents: List[RawDocument],
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """Aplica chunk_document a una lista completa de documentos."""
    all_chunks: List[Chunk] = []
    for doc in documents:
        doc_chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(doc_chunks)

    print(f"Generados {len(all_chunks)} chunks a partir de {len(documents)} documentos")
    return all_chunks


if __name__ == "__main__":
    # Uso rápido de prueba: python -m src.ingestion.chunker
    from src.ingestion.loader import load_documents

    docs = load_documents("data/raw")
    chunks = chunk_documents(docs)
    for c in chunks[:3]:
        print(f"\n--- {c.id} ---")
        print(c.text[:200] + "...")
