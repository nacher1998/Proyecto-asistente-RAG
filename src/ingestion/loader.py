"""
loader.py

Carga documentos desde disco (PDF, .txt, .md) y los normaliza a un
formato común antes de pasarlos al chunker.

Diseño: cada documento cargado se representa como un diccionario con
metadatos (fuente, tipo) además del texto. Esto es importante porque
más adelante, al citar las fuentes en la respuesta del LLM, necesitamos
saber de qué archivo vino cada fragmento.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class RawDocument:
    """Representa un documento crudo, antes de trocearlo en chunks."""
    source: str          # nombre o ruta del archivo de origen
    text: str            # texto completo extraído
    doc_type: str        # "pdf", "txt", "md"
    metadata: dict = field(default_factory=dict)


def load_pdf(path: Path) -> RawDocument:
    """Extrae el texto de un PDF, página por página."""
    reader = PdfReader(str(path))
    pages_text = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    full_text = "\n\n".join(pages_text)
    return RawDocument(
        source=path.name,
        text=full_text,
        doc_type="pdf",
        metadata={"num_pages": len(reader.pages)},
    )


def load_text_file(path: Path) -> RawDocument:
    """Carga un archivo de texto plano o markdown."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    doc_type = "md" if path.suffix.lower() == ".md" else "txt"
    return RawDocument(source=path.name, text=text, doc_type=doc_type)


def load_documents(input_dir: str) -> List[RawDocument]:
    """
    Recorre una carpeta y carga todos los documentos soportados.

    Args:
        input_dir: ruta a la carpeta con los documentos (p. ej. "data/raw")

    Returns:
        Lista de RawDocument, uno por archivo procesado con éxito.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"La carpeta {input_dir} no existe.")

    documents: List[RawDocument] = []
    supported_extensions = {".pdf": load_pdf, ".txt": load_text_file, ".md": load_text_file}

    for file_path in sorted(input_path.rglob("*")):
        if not file_path.is_file():
            continue

        loader_fn = supported_extensions.get(file_path.suffix.lower())
        if loader_fn is None:
            continue  # extensión no soportada, se ignora silenciosamente

        try:
            doc = loader_fn(file_path)
            if doc.text.strip():  # descarta documentos vacíos (p. ej. PDFs escaneados sin OCR)
                documents.append(doc)
            else:
                print(f"[aviso] {file_path.name} no tiene texto extraíble, se omite.")
        except Exception as e:
            print(f"[error] No se pudo procesar {file_path.name}: {e}")

    print(f"Cargados {len(documents)} documentos desde {input_dir}")
    return documents


if __name__ == "__main__":
    # Uso rápido de prueba: python -m src.ingestion.loader
    docs = load_documents("data/raw")
    for d in docs:
        print(f"- {d.source} ({d.doc_type}): {len(d.text)} caracteres")

