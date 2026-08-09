"""
scripts/evaluate.py

Corre un conjunto fijo de preguntas de prueba (evaluation/eval_questions.json)
contra el pipeline RAG completo y reporta cómo se comportó: si respondió
cuando debía, si se abstuvo cuando no había información, y si citó la
fuente esperada.

Esto convierte la fase de "probé el sistema a mano" en algo reproducible:
cada vez que cambies el chunking, el umbral de relevancia o el modelo,
puedes volver a correr este script y comparar resultados.

Uso:
    python scripts/evaluate.py
    python scripts/evaluate.py --questions evaluation/mis_preguntas.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.embedder import EmbeddingIndexer
from src.retrieval.retriever import Retriever
from src.generation.llm_client import LLMClient, answer_question


def load_questions(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_question(item: dict, retriever: Retriever, llm_client: LLMClient) -> dict:
    """Ejecuta una pregunta y compara el resultado contra lo esperado."""
    result = answer_question(item["question"], retriever, llm_client)

    # ¿el sistema respondió cuando debía, o se abstuvo cuando no debía?
    answered_correctly_abstained = (
        item["expect_answer"] == result.had_relevant_context
    )

    # ¿la fuente esperada aparece entre las citadas? (solo aplica si se esperaba respuesta)
    source_ok = True
    expected_source = item.get("expected_source_contains")
    if item["expect_answer"] and expected_source:
        source_ok = any(expected_source in s for s in result.sources)

    return {
        "id": item["id"],
        "question": item["question"],
        "expected_to_answer": item["expect_answer"],
        "did_answer": result.had_relevant_context,
        "abstention_correct": answered_correctly_abstained,
        "source_correct": source_ok,
        "answer": result.answer,
        "sources": result.sources,
        "passed": answered_correctly_abstained and source_ok,
    }


def main():
    parser = argparse.ArgumentParser(description="Evalúa el pipeline RAG contra un set de preguntas.")
    parser.add_argument(
        "--questions", default="evaluation/eval_questions.json", help="Ruta al archivo JSON de preguntas"
    )
    parser.add_argument(
        "--output-dir", default="evaluation/results", help="Carpeta donde guardar el reporte"
    )
    args = parser.parse_args()

    print(f"Cargando preguntas desde '{args.questions}'...")
    questions = load_questions(args.questions)

    print("Inicializando componentes del RAG...")
    indexer = EmbeddingIndexer()
    retriever = Retriever(indexer)
    llm_client = LLMClient()

    results = []
    for item in questions:
        print(f"\n→ [{item['id']}] {item['question']}")
        outcome = evaluate_question(item, retriever, llm_client)
        status = "✓ PASÓ" if outcome["passed"] else "✗ FALLÓ"
        print(f"  {status} — respondió: {outcome['did_answer']}, fuentes: {outcome['sources']}")
        results.append(outcome)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Resultado: {passed}/{total} preguntas pasaron ({passed/total:.0%})")
    print(f"{'='*50}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": f"{passed}/{total}", "results": results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nReporte detallado guardado en: {output_path}")


if __name__ == "__main__":
    main()
