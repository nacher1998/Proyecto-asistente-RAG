"""
Tests para el endpoint /ask de la API.

En lugar de usar el lifespan real (que descargaría el modelo de
embeddings y requeriría una API key), inyectamos directamente un
retriever y un llm_client simulados en app_state. Esto prueba la
lógica del endpoint (validación, manejo de errores, formato de
respuesta) de forma aislada y rápida.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.api.main import app, app_state
from src.retrieval.retriever import RetrievedChunk
from src.generation.llm_client import RAGAnswer


client = TestClient(app)


def setup_function():
    """Se ejecuta antes de cada test: inyectamos dependencias falsas."""
    fake_retriever = MagicMock()
    fake_retriever.retrieve.return_value = [
        RetrievedChunk(text="contenido", source="doc.txt", chunk_index=0, distance=0.1)
    ]

    fake_llm_client = MagicMock()

    app_state["retriever"] = fake_retriever
    app_state["llm_client"] = fake_llm_client


def teardown_function():
    app_state.clear()


def test_health_check_ok_when_ready():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_returns_answer_and_sources(monkeypatch):
    fake_answer = RAGAnswer(answer="Respuesta de prueba [Fragmento 1].", sources=["doc.txt"], had_relevant_context=True)
    monkeypatch.setattr("src.api.main.answer_question", lambda q, r, l: fake_answer)

    response = client.post("/ask", json={"question": "¿algo?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Respuesta de prueba [Fragmento 1]."
    assert data["sources"] == ["doc.txt"]
    assert data["had_relevant_context"] is True


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422  # error de validación de Pydantic


def test_ask_returns_503_when_not_ready():
    app_state.clear()  # simula que la app aún no terminó de inicializar
    response = client.post("/ask", json={"question": "¿algo?"})
    assert response.status_code == 503


def test_ask_returns_500_on_internal_error(monkeypatch):
    def raise_error(q, r, l):
        raise RuntimeError("fallo simulado")

    monkeypatch.setattr("src.api.main.answer_question", raise_error)

    response = client.post("/ask", json={"question": "¿algo?"})
    assert response.status_code == 500
