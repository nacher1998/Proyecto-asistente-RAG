"""
embedder.py

Convierte los chunks de texto en vectores (embeddings) y los indexa en
Chroma, una base de datos vectorial local y gratuita.

Decisión de diseño: usamos sentence-transformers en lugar de la API de
embeddings del proveedor de LLM por dos razones:
1. Es gratis y corre localmente, así no consumimos cuota de API solo
   para indexar (que puede ser miles de llamadas si el corpus crece).
2. Nos independiza del proveedor de LLM: podríamos cambiar de Claude a
   otro modelo para la generación sin tener que re-indexar todo el corpus.

El modelo elegido por defecto, "paraphrase-multilingual-MiniLM-L12-v2",
soporta español e inglés y es pequeño (rápido en CPU), lo cual importa
si vas a correr esto en tu portátil sin GPU.
"""

from pathlib import Path
from typing import List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from src.ingestion.chunker import Chunk

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_PERSIST_DIR = "chroma_db"
DEFAULT_COLLECTION_NAME = "rag_documents"


class EmbeddingIndexer:
    """
    Encapsula el modelo de embeddings y la colección de Chroma.

    Se usa tanto para construir el índice (build_index) como, más
    adelante, para hacer las búsquedas por similitud (ver retriever.py).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"embedding_model": model_name},
        )

    def build_index(self, chunks: List[Chunk], batch_size: int = 64) -> None:
        """
        Genera embeddings para una lista de chunks y los añade a Chroma.

        Se procesa en lotes (batch_size) porque generar embeddings de a
        uno es mucho más lento que en batch, y con corpus grandes la
        diferencia es notable.
        """
        if not chunks:
            print("[aviso] No hay chunks para indexar.")
            return

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]

            embeddings = self.model.encode(texts, show_progress_bar=False).tolist()

            self.collection.add(
                ids=[c.id for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {"source": c.source, "chunk_index": c.chunk_index, **c.metadata}
                    for c in batch
                ],
            )
            print(f"Indexados {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

        print(f"Índice completo: {self.collection.count()} chunks en total en la colección.")

    def query(self, query_text: str, top_k: int = 4) -> dict:
        """
        Busca los chunks más similares a una pregunta dada.

        Este método se usará desde retriever.py; se incluye aquí también
        porque es una forma rápida de probar que el índice quedó bien
        construido nada más terminar el embedding.
        """
        query_embedding = self.model.encode([query_text]).tolist()
        return self.collection.query(query_embeddings=query_embedding, n_results=top_k)

    def reset(self) -> None:
        """Borra la colección actual. Útil al re-indexar desde cero en desarrollo."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(name=self.collection.name)


def build_index_from_chunks(
    chunks: List[Chunk],
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
) -> EmbeddingIndexer:
    """Función de conveniencia: crea el indexer y construye el índice en un paso."""
    indexer = EmbeddingIndexer(
        model_name=model_name,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    indexer.build_index(chunks)
    return indexer


if __name__ == "__main__":
    # Uso rápido de prueba: python -m src.embeddings.embedder
    from src.ingestion.loader import load_documents
    from src.ingestion.chunker import chunk_documents

    docs = load_documents("data/raw")
    chunks = chunk_documents(docs)
    indexer = build_index_from_chunks(chunks)

    # prueba rápida de que el índice responde
    results = indexer.query("pregunta de ejemplo sobre el corpus", top_k=3)
    print(results)

