"""Оценка качества извлечения триплетов: precision / recall / F1 по рёбрам.

Использует размеченные пары (текст -> эталонные триплеты). Два набора:

  * gold      — held-out набор (klubok/eval/gold_set.py), НЕ входит в few-shot
                промпт. Честная оценка. Используется по умолчанию.
  * fewshot   — те же пары, что в промпте (FEW_SHOT_EXAMPLES), но с leave-one-out:
                оцениваемый пример исключается из few-shot, чтобы не было утечки.

Запуск:
    python scripts/eval_extraction.py                 # gold-набор
    python scripts/eval_extraction.py --dataset fewshot
    python scripts/eval_extraction.py --no-few-shot   # zero-shot режим

Бэкенд LLM берётся из .env (LLM_BACKEND). На MockLLM метрики будут низкими —
это нормально: реальная оценка делается с LLM_BACKEND=metalgpt на машине с GPU.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401,E402  (путь к пакету + UTF-8 вывод)

from config import settings
from klubok.extraction.prompts import (
    FEW_SHOT_EXAMPLES, EXTRACTION_SYSTEM, build_extraction_prompt,
)
from klubok.extraction.extractor import parse_extraction
from klubok.extraction.llm_client import get_llm
from klubok.eval.gold_set import GOLD_SET
from klubok.eval.metrics import (
    score_extraction, micro_average, gold_from_dict, ExampleScore,
)


def _prompt_for(text: str, idx: int, dataset: str, few_shot: bool) -> str:
    """Собрать промпт; для fewshot-набора делаем leave-one-out."""
    if not few_shot:
        return build_extraction_prompt(text, few_shot=False)
    if dataset == "fewshot":
        examples = [ex for j, ex in enumerate(FEW_SHOT_EXAMPLES) if j != idx]
        return build_extraction_prompt(text, few_shot=True, examples=examples)
    return build_extraction_prompt(text, few_shot=True)   # gold: используем все примеры


def evaluate(dataset: str, few_shot: bool) -> None:
    data = GOLD_SET if dataset == "gold" else FEW_SHOT_EXAMPLES
    llm = get_llm()
    scores: list[ExampleScore] = []

    print(f"Бэкенд LLM: {settings.llm_backend} | набор: {dataset} | "
          f"few-shot: {few_shot} | примеров: {len(data)}\n")
    if settings.llm_backend == "mock":
        print("⚠️  MockLLM: метрики низкие по построению. Для реальной оценки — "
              "LLM_BACKEND=yandex (или metalgpt).\n")

    header = f"{'#':>2}  {'ent P/R/F1':>16}  {'rel P/R/F1':>16}  {'attr P/R/F1':>16}"
    print(header)
    print("-" * len(header))

    for idx, ex in enumerate(data):
        prompt = _prompt_for(ex["text"], idx, dataset, few_shot)
        raw = llm.complete(prompt, system=EXTRACTION_SYSTEM)
        pred = parse_extraction(raw, doc_id=f"eval_{idx}", chunk_id=None)
        gold = gold_from_dict(ex["output"], doc_id=f"eval_{idx}")

        s = score_extraction(pred, gold)
        scores.append(s)
        e, r, a = s.entities, s.relations, s.attributes
        print(f"{idx:>2}  "
              + f"{e.precision:.2f}/{e.recall:.2f}/{e.f1:.2f}".rjust(16) + "  "
              + f"{r.precision:.2f}/{r.recall:.2f}/{r.f1:.2f}".rjust(16) + "  "
              + f"{a.precision:.2f}/{a.recall:.2f}/{a.f1:.2f}".rjust(16))

        if s.missed_relations:
            for m in s.missed_relations[:5]:
                print(f"      ✗ не нашли ребро: {m[0]} -{m[1]}-> {m[2]}")
        if s.spurious_relations:
            for m in s.spurious_relations[:5]:
                print(f"      + лишнее ребро : {m[0]} -{m[1]}-> {m[2]}")
        if s.wrong_attributes:
            for m in s.wrong_attributes[:5]:
                print(f"      ≠ атрибут      : {m[0]} · {m[1]}={m[2]}")

    print("\n=== Микро-усреднение по всему набору ===")
    agg = micro_average(scores)
    for level in ("entities", "relations", "attributes"):
        m = agg[level]
        print(f"  {level:10} P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  (tp={m['tp']} fp={m['fp']} fn={m['fn']})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Оценка извлечения триплетов")
    ap.add_argument("--dataset", choices=["gold", "fewshot"], default="gold")
    ap.add_argument("--no-few-shot", dest="few_shot", action="store_false",
                    help="zero-shot режим (без примеров в промпте)")
    args = ap.parse_args()
    evaluate(args.dataset, args.few_shot)


if __name__ == "__main__":
    main()
