"""
main.py

Expone el pipeline completo (retrieval + generación) como una API REST
con FastAPI.

Decisión de diseño: el indexer, retriever y llm_client se inicializan
UNA SOLA VEZ al arrancar la aplicación (mediante el lifespan de FastAPI),
no en cada request. Esto es importante porque cargar el modelo de
embeddings y conectar con Chroma tiene un costo de arranque; hacerlo en
cada petición sería lentísimo e innecesario.
"""

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.embeddings.embedder import EmbeddingIndexer
from src.retrieval.retriever import Retriever
from src.generation.llm_client import LLMClient, answer_question


# --- Estado de la aplicación, inicializado una sola vez ---

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Se ejecuta al arrancar y al apagar la aplicación (patrón oficial de
    FastAPI para reemplazar los antiguos @app.on_event("startup")).
    """
    print("Inicializando componentes del RAG (esto puede tardar unos segundos)...")
    indexer = EmbeddingIndexer()
    app_state["retriever"] = Retriever(indexer)
    app_state["llm_client"] = LLMClient()
    print("Componentes listos. API disponible.")

    yield  # la aplicación corre aquí

    app_state.clear()
    print("Aplicación cerrada, estado limpiado.")


app = FastAPI(
    title="RAG Assistant API",
    description="API para hacer preguntas sobre un corpus de documentos usando RAG.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Esquemas de request / response ---

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Pregunta del usuario en lenguaje natural")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Número de fragmentos a recuperar (opcional)")


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    had_relevant_context: bool


# --- Endpoints ---

@app.get("/health")
def health_check():
    """Endpoint simple para verificar que la API está viva y los componentes cargados."""
    is_ready = "retriever" in app_state and "llm_client" in app_state
    return {"status": "ok" if is_ready else "initializing"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Recibe una pregunta y devuelve la respuesta generada junto con las
    fuentes del corpus que se usaron para construirla.
    """
    if "retriever" not in app_state:
        raise HTTPException(status_code=503, detail="El servicio aún se está inicializando.")

    retriever = app_state["retriever"]
    llm_client = app_state["llm_client"]

    if request.top_k:
        retriever.top_k = request.top_k

    try:
        result = answer_question(request.question, retriever, llm_client)
    except Exception as e:
        # No exponemos el detalle interno del error al cliente por seguridad,
        # pero sí lo registramos en el log del servidor para poder depurar.
        print(f"[error] Fallo al procesar la pregunta: {e}")
        raise HTTPException(status_code=500, detail="Error interno al procesar la pregunta.")

    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        had_relevant_context=result.had_relevant_context,
    )


if __name__ == "__main__":
    # Para desarrollo local: python -m src.api.main
    # En producción se recomienda: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

