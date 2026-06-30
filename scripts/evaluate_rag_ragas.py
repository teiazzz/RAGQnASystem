from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.rag_tokenizer import tokenize  # noqa: E402


DEFAULT_EVAL_FILE = PROJECT_ROOT / "data" / "rag_eval" / "rag_eval_cases.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "rag_eval" / "reports"


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    reference_answer: str
    expected_sources: list[str]
    expected_keywords: list[str]
    answer_keywords: list[str]
    kg_knowledge: list[str]


@dataclass
class CaseResult:
    case: EvalCase
    sources: list[Any]
    retrieval_rank: int | None
    latency_ms: float
    answer: str = ""


def load_eval_cases(path: Path, max_cases: int | None = None) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw = json.loads(line)
            cases.append(
                EvalCase(
                    case_id=str(raw.get("id") or line_no),
                    query=str(raw["query"]),
                    reference_answer=str(raw.get("reference_answer") or ""),
                    expected_sources=_as_str_list(raw.get("expected_sources")),
                    expected_keywords=_as_str_list(raw.get("expected_keywords")),
                    answer_keywords=_as_str_list(raw.get("answer_keywords")),
                    kg_knowledge=_as_str_list(raw.get("kg_knowledge")),
                )
            )
            if max_cases is not None and len(cases) >= max_cases:
                break
    if not cases:
        raise ValueError(f"No eval cases loaded from {path}")
    return cases


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if str(item).strip()]


def source_text(source: Any) -> str:
    return "\n".join(
        [
            str(getattr(source, "source_type", "")),
            str(getattr(source, "source_title", "")),
            str(getattr(source, "section", "")),
            str(getattr(source, "content", "")),
        ]
    )


def source_matches(source: Any, case: EvalCase) -> bool:
    haystack = source_text(source)
    if any(expected and expected in haystack for expected in case.expected_sources):
        return True
    return any(keyword and keyword in haystack for keyword in case.expected_keywords)


def hit_rank(sources: list[Any], case: EvalCase) -> int | None:
    for idx, source in enumerate(sources, start=1):
        if source_matches(source, case):
            return idx
    return None


def context_precision(sources: list[Any], case: EvalCase) -> float:
    relevant = 0
    precision_sum = 0.0
    for rank, source in enumerate(sources, start=1):
        if not source_matches(source, case):
            continue
        relevant += 1
        precision_sum += relevant / rank
    if relevant == 0:
        return 0.0
    return precision_sum / relevant


def context_recall(sources: list[Any], case: EvalCase) -> float:
    keywords = _dedupe([*case.expected_keywords, *case.answer_keywords])
    if not keywords:
        return 1.0 if hit_rank(sources, case) is not None else 0.0
    context = "\n".join(source_text(source) for source in sources)
    return _keyword_coverage(context, keywords)


def answer_relevancy_proxy(answer: str, case: EvalCase) -> float | None:
    if not answer:
        return None
    keywords = case.answer_keywords or case.expected_keywords
    if keywords:
        return _keyword_coverage(answer, keywords)
    query_tokens = set(tokenize(case.query))
    answer_tokens = set(tokenize(answer))
    return len(query_tokens & answer_tokens) / max(len(query_tokens), 1)


def faithfulness_proxy(answer: str, sources: list[Any], case: EvalCase) -> float | None:
    if not answer:
        return None
    context = "\n".join(source_text(source) for source in sources)
    keywords = case.answer_keywords or case.expected_keywords
    answer_hits = [keyword for keyword in keywords if keyword and keyword in answer]
    if answer_hits:
        return _keyword_coverage(context, answer_hits)
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = set(_content_tokens(context))
    return len(set(answer_tokens) & context_tokens) / max(len(set(answer_tokens)), 1)


def citation_precision(answer: str, sources: list[Any]) -> float | None:
    if not answer:
        return None
    citations = [int(item) for item in re.findall(r"\[(\d+)\]", answer)]
    if not citations:
        return 0.0
    valid_ids = {int(getattr(source, "citation_id", 0) or 0) for source in sources}
    valid = sum(1 for citation in citations if citation in valid_ids)
    return valid / len(citations)


def _content_tokens(text: str) -> list[str]:
    return [
        token
        for token in tokenize(text)
        if len(token) >= 2 or re.fullmatch(r"[a-zA-Z0-9]+", token)
    ]


def _keyword_coverage(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword and keyword in text)
    return hits / len(keywords)


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def brief_sources(sources: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for source in sources:
        rows.append(
            {
                "citation_id": getattr(source, "citation_id", None),
                "source_type": getattr(source, "source_type", ""),
                "source_title": getattr(source, "source_title", ""),
                "section": getattr(source, "section", ""),
                "authority_level": getattr(source, "authority_level", ""),
                "score": round(
                    float(
                        getattr(source, "rerank_score", 0.0)
                        or getattr(source, "fused_score", 0.0)
                    ),
                    4,
                ),
                "content_preview": str(getattr(source, "content", ""))[:220],
            }
        )
    return rows


def build_generation_prompt(case: EvalCase, sources: list[Any]) -> str:
    contexts = []
    for source in sources:
        citation_id = int(getattr(source, "citation_id", 0) or 0)
        contexts.append(
            f"[{citation_id}] {getattr(source, 'source_title', '')} / "
            f"{getattr(source, 'section', '')}: "
            f"{str(getattr(source, 'content', ''))[:900]}"
        )
    return (
        "<指令>你是医疗健康问答助手。只能依据给定编号来源回答，"
        "不要编造来源；使用来源信息时在句尾标注引用编号。"
        "如果来源不足以回答，请直接说明证据不足并建议咨询医生。"
        "回答控制在 120 字以内。</指令>\n"
        f"用户问题：{case.query}\n"
        "编号来源：\n"
        + "\n".join(contexts)
        + "\n回答："
    )


async def generate_answer(case: EvalCase, sources: list[Any]) -> str:
    from app.services import llm_service

    return (await llm_service.generate(build_generation_prompt(case, sources), temperature=0.0)).strip()


async def maybe_index_corpus(args: argparse.Namespace) -> None:
    if not args.index_corpus:
        return
    from app.db.session import async_session_maker
    from app.services.corpus_indexer import index_medical_corpus

    async with async_session_maker() as session:
        result = await index_medical_corpus(
            session,
            limit=args.index_limit,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
    print(
        "indexed corpus: "
        f"records_seen={result.records_seen} created={result.created_chunks} "
        f"skipped={result.skipped_chunks}"
    )


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from sqlalchemy import func, select

    from app.db.models import DocumentChunk
    from app.db.session import async_session_maker, engine, init_db
    from app.services.hybrid_retriever import get_hybrid_retriever

    if args.init_db:
        await init_db()
    await maybe_index_corpus(args)

    cases = load_eval_cases(args.eval_file, max_cases=args.max_cases)
    retriever = get_hybrid_retriever()
    results: list[CaseResult] = []

    try:
        async with async_session_maker() as session:
            chunk_count = await session.scalar(select(func.count()).select_from(DocumentChunk))
            if not chunk_count:
                print(
                    "WARNING: document_chunks is empty. Run with --index-corpus or index documents first.",
                    file=sys.stderr,
                )

            for case in cases:
                start = time.perf_counter()
                sources = await retriever.search(
                    session,
                    query=case.query,
                    kg_knowledge=case.kg_knowledge,
                    top_k=args.top_k,
                    candidate_k=args.candidate_k,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                result = CaseResult(
                    case=case,
                    sources=sources,
                    retrieval_rank=hit_rank(sources, case),
                    latency_ms=latency_ms,
                )
                if args.generate_answers:
                    result.answer = await generate_answer(case, sources)
                results.append(result)

        summary = build_summary(args, results)
        output_paths = write_outputs(args, summary, results)
        summary["artifacts"] = output_paths
        if args.use_ragas:
            summary["ragas"] = try_run_ragas(output_paths["ragas_input"])
        return summary
    finally:
        await engine.dispose()


def build_summary(args: argparse.Namespace, results: list[CaseResult]) -> dict[str, Any]:
    total = len(results)
    retrieval_hits = sum(1 for item in results if item.retrieval_rank is not None)
    reciprocal_rank_sum = sum(
        1 / item.retrieval_rank for item in results if item.retrieval_rank is not None
    )
    context_precision_scores = [context_precision(item.sources, item.case) for item in results]
    context_recall_scores = [context_recall(item.sources, item.case) for item in results]
    latencies = [item.latency_ms for item in results]

    answer_relevancy_scores = [
        score
        for item in results
        if (score := answer_relevancy_proxy(item.answer, item.case)) is not None
    ]
    faithfulness_scores = [
        score
        for item in results
        if (score := faithfulness_proxy(item.answer, item.sources, item.case)) is not None
    ]
    citation_scores = [
        score
        for item in results
        if (score := citation_precision(item.answer, item.sources)) is not None
    ]

    summary: dict[str, Any] = {
        "eval_file": str(args.eval_file),
        "cases": total,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "generation_enabled": args.generate_answers,
        "metrics": {
            "retrieval_recall_at_k": round(retrieval_hits / total, 4),
            "retrieval_mrr": round(reciprocal_rank_sum / total, 4),
            "context_precision": round(_mean(context_precision_scores), 4),
            "context_recall": round(_mean(context_recall_scores), 4),
            "avg_latency_ms": round(_mean(latencies), 2),
        },
        "failures": build_failures(results),
    }
    if args.generate_answers:
        faithfulness = _mean(faithfulness_scores)
        summary["metrics"].update(
            {
                "answer_relevancy_proxy": round(_mean(answer_relevancy_scores), 4),
                "faithfulness_proxy": round(faithfulness, 4),
                "citation_precision": round(_mean(citation_scores), 4),
                "hallucination_rate_proxy": round(1 - faithfulness, 4),
            }
        )
    return summary


def build_failures(results: list[CaseResult]) -> list[dict[str, Any]]:
    failures = []
    for item in results:
        answer_rel = answer_relevancy_proxy(item.answer, item.case)
        faithful = faithfulness_proxy(item.answer, item.sources, item.case)
        if (
            item.retrieval_rank is not None
            and (answer_rel is None or answer_rel >= 0.5)
            and (faithful is None or faithful >= 0.5)
        ):
            continue
        failures.append(
            {
                "id": item.case.case_id,
                "query": item.case.query,
                "retrieval_rank": item.retrieval_rank,
                "answer_relevancy_proxy": answer_rel,
                "faithfulness_proxy": faithful,
                "expected_sources": item.case.expected_sources,
                "expected_keywords": item.case.expected_keywords,
                "answer_keywords": item.case.answer_keywords,
                "answer": item.answer,
                "top_sources": brief_sources(item.sources),
            }
        )
    return failures


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def write_outputs(
    args: argparse.Namespace, summary: dict[str, Any], results: list[CaseResult]
) -> dict[str, str]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.output_dir / f"rag_eval_report_{timestamp}.json"
    ragas_input_path = args.output_dir / f"ragas_input_{timestamp}.jsonl"

    rows = [case_result_to_dict(item) for item in results]
    payload = {**summary, "case_results": rows}
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with ragas_input_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {
                        "question": row["query"],
                        "answer": row["answer"],
                        "contexts": row["contexts"],
                        "ground_truth": row["reference_answer"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {"report": str(report_path), "ragas_input": str(ragas_input_path)}


def case_result_to_dict(item: CaseResult) -> dict[str, Any]:
    return {
        "id": item.case.case_id,
        "query": item.case.query,
        "reference_answer": item.case.reference_answer,
        "answer": item.answer,
        "retrieval_rank": item.retrieval_rank,
        "context_precision": round(context_precision(item.sources, item.case), 4),
        "context_recall": round(context_recall(item.sources, item.case), 4),
        "answer_relevancy_proxy": _round_optional(
            answer_relevancy_proxy(item.answer, item.case)
        ),
        "faithfulness_proxy": _round_optional(
            faithfulness_proxy(item.answer, item.sources, item.case)
        ),
        "citation_precision": _round_optional(citation_precision(item.answer, item.sources)),
        "latency_ms": round(item.latency_ms, 2),
        "sources": brief_sources(item.sources),
        "contexts": [str(getattr(source, "content", "")) for source in item.sources],
    }


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def try_run_ragas(ragas_input_path: str) -> dict[str, Any]:
    """可选真实 RAGAS 调用；未安装或未配置 judge LLM 时不会影响本地评测。"""
    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate as ragas_evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_relevancy,
            context_precision as ragas_context_precision,
            context_recall as ragas_context_recall,
            faithfulness,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "unavailable",
            "reason": f"ragas/datasets not installed: {exc}",
        }

    try:
        rows = [
            json.loads(line)
            for line in Path(ragas_input_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        dataset = Dataset.from_list(rows)
        result = ragas_evaluate(
            dataset,
            metrics=[
                ragas_context_precision,
                ragas_context_recall,
                faithfulness,
                answer_relevancy,
            ],
        )
        return {"status": "ok", "scores": dict(result)}
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "reason": str(exc),
            "hint": "RAGAS usually needs a configured judge LLM. Use the exported ragas_input JSONL if you run RAGAS elsewhere.",
        }


def print_summary(summary: dict[str, Any]) -> None:
    print(
        f"eval_file={summary['eval_file']} cases={summary['cases']} "
        f"top_k={summary['top_k']} candidate_k={summary['candidate_k']} "
        f"generation={summary['generation_enabled']}"
    )
    for key, value in summary["metrics"].items():
        print(f"{key:<28} {value}")
    print(f"failures                    {len(summary.get('failures', []))}")
    print("artifacts:")
    for key, value in summary.get("artifacts", {}).items():
        print(f"  {key}: {value}")
    if "ragas" in summary:
        print("ragas:")
        print(json.dumps(summary["ragas"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval and RAGAS-style answer quality metrics."
    )
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--init-db", action="store_true")
    parser.add_argument("--index-corpus", action="store_true")
    parser.add_argument("--index-limit", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--generate-answers", action="store_true")
    parser.add_argument("--use-ragas", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = asyncio.run(evaluate(args))
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_summary(summary)


if __name__ == "__main__":
    main()
