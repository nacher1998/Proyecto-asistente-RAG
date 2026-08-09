"""
streamlit_app.py

Interfaz de chat simple que consume el endpoint /ask de la API de
FastAPI (no llama directamente al pipeline RAG).

Decisión de diseño: la interfaz habla con la API por HTTP, no importa
directamente los módulos de src/. Esto mantiene la separación entre
frontend y backend — en un proyecto real, la API podría estar corriendo
en otro servidor, y la interfaz ni se entera de los detalles internos
del RAG (embeddings, Chroma, etc.).

Para correr la app completa se necesitan dos procesos en paralelo:
  1. uvicorn src.api.main:app --reload          (backend, puerto 8000)
  2. streamlit run app/streamlit_app.py         (frontend, puerto 8501)
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")


def query_api(question: str, api_url: str = API_URL, timeout: int = 30) -> dict:
    """
    Envía la pregunta al endpoint /ask y devuelve la respuesta parseada.

    Extraída como función independiente (en vez de código suelto en el
    script) para poder testearla con requests simulado, sin necesitar
    Streamlit corriendo de verdad.
    """
    response = requests.post(
        f"{api_url}/ask",
        json={"question": question},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def render_answer(result: dict) -> None:
    """Muestra la respuesta y, si las hay, las fuentes usadas."""
    st.markdown(result["answer"])

    if result.get("sources"):
        with st.expander(f"📄 Fuentes consultadas ({len(result['sources'])})"):
            for source in result["sources"]:
                st.markdown(f"- `{source}`")


def main():
    st.set_page_config(page_title="Asistente RAG", page_icon="🤖")
    st.title("🤖 Asistente sobre el corpus")
    st.caption(
        "Este asistente responde únicamente con información de los documentos indexados. "
        "Si no encuentra la respuesta en el corpus, te lo dirá en vez de inventar."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar el historial de la conversación
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "sources" in message:
                render_answer(message)
            else:
                st.markdown(message["content"])

    # Input del usuario
    user_question = st.chat_input("Escribe tu pregunta...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Buscando en el corpus..."):
                try:
                    result = query_api(user_question)
                    render_answer(result)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "answer": result["answer"],
                            "sources": result.get("sources", []),
                        }
                    )
                except requests.exceptions.ConnectionError:
                    error_msg = (
                        "⚠️ No se pudo conectar con la API. "
                        "Verifica que esté corriendo en "
                        f"`{API_URL}` (uvicorn src.api.main:app --reload)."
                    )
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except requests.exceptions.HTTPError as e:
                    error_msg = f"⚠️ Error del servidor: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    with st.sidebar:
        st.header("Acerca de este proyecto")
        st.markdown(
            "Asistente construido con RAG (Retrieval-Augmented Generation): "
            "recupera fragmentos relevantes de un corpus propio y los usa "
            "como contexto para generar respuestas verificables."
        )
        if st.button("🗑️ Limpiar conversación"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()

