# 🤖 Asistente RAG sobre documentación propia

Un asistente de preguntas y respuestas que combina **recuperación de información (retrieval)** con un **modelo de lenguaje (LLM)** para responder preguntas basándose únicamente en un corpus de documentos propio — no en el conocimiento general del modelo.

> Proyecto construido como parte de mi transición hacia un rol de Datos/IA, para demostrar comprensión de punta a punta de un pipeline RAG: desde la ingesta de datos hasta el despliegue como API con interfaz de usuario.

---

## 🎯 El problema que resuelve

Los LLMs generales no conocen documentos privados o muy específicos (normativa interna, documentación técnica de un producto, políticas de una empresa) y tienden a **inventar respuestas con seguridad** cuando no saben algo. Este proyecto resuelve eso mediante RAG: en lugar de confiar en la memoria del modelo, primero busca los fragmentos más relevantes en el corpus y se los da como contexto, instruyendo al modelo a **decir explícitamente cuando no tiene información suficiente**.

## 🏗️ Arquitectura

```
Pregunta del usuario
        │
        ▼
┌───────────────────┐
│   Streamlit UI     │  (app/streamlit_app.py)
└─────────┬──────────┘
          │ HTTP POST /ask
          ▼
┌───────────────────┐
│   FastAPI (API)     │  (src/api/main.py)
└─────────┬──────────┘
          │
          ▼
┌───────────────────┐      ┌──────────────────────┐
│   Retriever         │ ──▶  │  Chroma (base vectorial) │
│ (src/retrieval)     │      └──────────────────────┘
└─────────┬──────────┘
          │ fragmentos relevantes
          ▼
┌───────────────────┐
│   LLM Client        │ ──▶  API de Claude
│ (src/generation)    │
└─────────┬──────────┘
          │
          ▼
   Respuesta + fuentes citadas
```

**Pipeline offline (indexación, se corre una vez o cuando cambia el corpus):**

```
data/raw/*.pdf,*.txt  →  loader.py  →  chunker.py  →  embedder.py  →  Chroma
```

## 🧰 Stack técnico

| Componente          | Tecnología                                  | Por qué                                                      |
|---------------------|----------------------------------------------|---------------------------------------------------------------|
| LLM                  | API de Claude (Anthropic)                    | Buena calidad de razonamiento y seguimiento de instrucciones   |
| Embeddings           | sentence-transformers (local, gratis)        | No depende del proveedor del LLM; sin costo por token indexado |
| Base vectorial       | Chroma                                       | Ligera, local, sin infraestructura adicional                   |
| Backend              | FastAPI                                      | Async, validación automática con Pydantic, estándar en la industria |
| Frontend             | Streamlit                                    | Interfaz de chat funcional en poco código                      |
| Tests                | pytest                                        | Suite rápida con dependencias externas simuladas (mocks)       |

## 📂 Estructura del proyecto

```
rag-assistant/
├── src/
│   ├── ingestion/       # carga y trocea documentos
│   ├── embeddings/      # genera embeddings e indexa en Chroma
│   ├── retrieval/       # búsqueda por similitud + filtro de relevancia
│   ├── generation/      # prompts + llamada al LLM
│   └── api/             # endpoint FastAPI
├── app/                 # interfaz de Streamlit
├── scripts/             # indexación y evaluación
├── tests/               # suite de pytest
├── evaluation/          # preguntas de prueba y resultados
└── data/raw/             # documentos fuente (no versionados)
```

## 🚀 Cómo ejecutarlo

### 1. Requisitos previos
- Python 3.10+
- Una clave de API de [Anthropic](https://console.anthropic.com/)

### 2. Instalación

```bash
git clone https://github.com/tu-usuario/rag-assistant.git
cd rag-assistant
python -m venv venv
source venv/bin/activate  # en Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # y añade tu ANTHROPIC_API_KEY
```

### 3. Añade tus documentos

Coloca tus archivos `.pdf`, `.txt` o `.md` en `data/raw/`.

### 4. Construye el índice

```bash
python scripts/build_index.py
```

### 5. Levanta el backend y el frontend (en dos terminales)

```bash
# Terminal 1: API
uvicorn src.api.main:app --reload

# Terminal 2: Interfaz
streamlit run app/streamlit_app.py
```

Abre `http://localhost:8501` y empieza a preguntar.

### 6. Corre los tests

```bash
pytest tests/ -v
```

## 🧠 Decisiones de diseño (y sus trade-offs)

- **Embeddings locales vs. API de embeddings del proveedor del LLM**: elegí `sentence-transformers` corriendo localmente para no depender de una sola API y evitar costo por cada chunk indexado, a cambio de un pequeño costo de tiempo de carga del modelo al iniciar.
- **Umbral de relevancia en el retriever**: en vez de pasar siempre los top-k fragmentos al LLM sin importar qué tan relevantes sean, se descartan los que superan un umbral de distancia. Esto reduce alucinaciones cuando la pregunta está fuera del dominio del corpus, a costa de que el umbral deba calibrarse por corpus (ver sección de evaluación).
- **Chunking por palabras con solapamiento** en vez de por oraciones o por tokens exactos: es más simple de implementar y suficientemente bueno para este caso de uso, aunque un enfoque semántico (chunking por párrafos o secciones) podría mejorar la calidad del retrieval en corpus muy estructurados.
- **Componentes inicializados una sola vez** (`lifespan` de FastAPI) en vez de por request: evita recargar el modelo de embeddings en cada pregunta, a cambio de un mayor uso de memoria mientras el servidor está activo.

## ⚠️ Limitaciones conocidas

- El umbral de relevancia (`0.7` por defecto) es un punto de partida, no un valor calibrado — cada corpus tiene una distribución de distancias distinta.
- El chunking por palabras puede cortar frases a la mitad; funciona razonablemente bien pero no es semánticamente óptimo.
- No hay manejo de PDFs escaneados sin OCR (el loader los omite con un aviso).
- La evaluación (ver abajo) se hizo con un conjunto pequeño de preguntas; un corpus en producción necesitaría un set de evaluación más grande y variado.

## 📊 Evaluación

Se probó el sistema con un conjunto de preguntas dentro y fuera de dominio (`evaluation/eval_questions.json`), verificando:
- Que responda correctamente preguntas cuya respuesta está en el corpus.
- Que reconozca honestamente cuando no tiene información suficiente.
- Que cite las fuentes correctas.

*(Completar aquí con los resultados reales una vez corrido `scripts/evaluate.py` sobre tu propio corpus.)*

## 🔮 Próximos pasos

- Chunking semántico (por párrafos/secciones en vez de por conteo de palabras)
- Re-ranking de resultados con un modelo cross-encoder antes de pasar el contexto al LLM
- Caché de preguntas frecuentes para reducir llamadas al LLM
- Despliegue del backend en un servicio gratuito (Render, Railway) y del frontend en Streamlit Cloud

## 📝 Licencia

MIT
