"""
scripts/build_index.py

Punto de entrada para construir (o reconstruir) el índice vectorial a
partir de los documentos en data/raw/. Se ejecuta una vez al principio
y cada vez que el corpus cambie.

Uso:
    python scripts/build_index.py
    python scripts/build_index.py --reset          # borra el índice existente antes de reindexar
    python scripts/build_index.py --input-dir otra_carpeta/
"""

import argparse
import sys
from pathlib import Path

# Permite ejecutar el script directamente (python scripts/build_index.py)
# sin tener que instalar el proyecto como paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.embeddings.embedder import EmbeddingIndexer


def main():
    parser = argparse.ArgumentParser(description="Construye el índice vectorial del corpus.")
    parser.add_argument(
        "--input-dir", default="data/raw", help="Carpeta con los documentos fuente (default: data/raw)"
    )
    parser.add_argument(
        "--persist-dir", default="chroma_db", help="Carpeta donde persiste Chroma (default: chroma_db)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="Tamaño de cada chunk en palabras (default: 500)"
    )
    parser.add_argument(
        "--overlap", type=int, default=50, help="Palabras de solapamiento entre chunks (default: 50)"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Borra el índice existente antes de reindexar"
    )
    args = parser.parse_args()

    print(f"1/3 — Cargando documentos desde '{args.input_dir}'...")
    documents = load_documents(args.input_dir)

    if not documents:
        print(
            f"No se encontraron documentos en '{args.input_dir}'. "
            "Añade archivos .pdf, .txt o .md antes de continuar."
        )
        sys.exit(1)

    print(f"2/3 — Troceando documentos (chunk_size={args.chunk_size}, overlap={args.overlap})...")
    chunks = chunk_documents(documents, chunk_size=args.chunk_size, overlap=args.overlap)

    print(f"3/3 — Generando embeddings e indexando en '{args.persist_dir}'...")
    indexer = EmbeddingIndexer(persist_directory=args.persist_dir)

    if args.reset:
        print("Flag --reset activado: borrando índice existente antes de reindexar.")
        indexer.reset()

    indexer.build_index(chunks)

    print("\n✓ Índice construido con éxito. Ya puedes levantar la API con:")
    print("  uvicorn src.api.main:app --reload")


if __name__ == "__main__":
    main()

