"""
prompts.py

Plantillas de prompts para la fase de generación del RAG.

Decisiones de diseño clave (explícitas a propósito, para poder
justificarlas en una entrevista):

1. El system prompt instruye explícitamente al modelo a NO responder
   si el contexto no contiene la información — esto es la defensa
   principal contra alucinaciones, más importante incluso que el
   umbral de relevancia del retriever.

2. Se le pide citar la fuente de cada afirmación (p. ej. "[Fragmento 1]")
   para que la respuesta sea verificable por el usuario, un requisito
   típico en RAGs de dominios donde la precisión importa (legal, salud,
   normativa interna, etc.).

3. Separamos claramente system prompt (instrucciones de comportamiento)
   de user prompt (pregunta + contexto), siguiendo la práctica estándar
   de la API de mensajes de Claude.
"""


SYSTEM_PROMPT = """Eres un asistente que responde preguntas basándote ÚNICAMENTE en los \
fragmentos de información proporcionados en cada consulta.

Reglas estrictas que debes seguir:
1. Responde SOLO con información que esté explícitamente en los fragmentos proporcionados.
2. Si los fragmentos no contienen información suficiente para responder, di claramente: \
"No tengo información suficiente en el corpus para responder a esto." No inventes ni \
completes con conocimiento externo.
3. Cuando uses información de un fragmento, cita su número entre corchetes, por ejemplo: \
"El plazo es de 15 días [Fragmento 1]."
4. Sé conciso y directo. No repitas la pregunta ni añadas relleno innecesario.
5. Si distintos fragmentos se contradicen, señala la contradicción en vez de elegir uno \
arbitrariamente."""


def build_user_prompt(query: str, formatted_context: str) -> str:
    """
    Construye el prompt de usuario combinando la pregunta con el contexto
    recuperado (ya formateado por retriever.format_chunks_for_prompt).
    """
    return f"""Contexto disponible:

{formatted_context}

Pregunta del usuario: {query}

Responde siguiendo estrictamente las reglas indicadas en las instrucciones."""


if __name__ == "__main__":
    # Ejemplo rápido de cómo queda el prompt final
    ejemplo_contexto = "[Fragmento 1 — fuente: politica.txt]\nLos reclamos se resuelven en 15 días."
    print("--- SYSTEM PROMPT ---")
    print(SYSTEM_PROMPT)
    print("\n--- USER PROMPT ---")
    print(build_user_prompt("¿cuánto tarda un reclamo?", ejemplo_contexto))

