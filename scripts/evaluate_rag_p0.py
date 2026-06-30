from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    expected_sources: list[str]
    expected_keywords: list[str]
    kg_knowledge: list[str]


@dataclass
class MethodStats:
    hits: int = 0
    reciprocal_rank_sum: float = 0.0
    latency_ms: list[float] | None = None

    def __post_init__(self) -> None:
        if self.latency_ms is None:
            self.latency_ms = []


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw = json.loads(line)
            expected_sources = raw.get("expected_sources") or raw.get("expected_source") or []
            if isinstance(expected_sources, str):
                expected_sources = [expected_sources]
            expected_keywords = raw.get("expected_keywords") or []
            if isinstance(expected_keywords, str):
                expected_keywords = [expected_keywords]
            kg_knowledge = raw.get("kg_knowledge") or []
            if isinstance(kg_knowledge, str):
                kg_knowledge = [kg_knowledge]
            cases.append(
                EvalCase(
                    case_id=str(raw.get("id") or line_no),
                    query=str(raw["query"]),
                    expected_sources=[str(item) for item in expected_sources],
                    expected_keywords=[str(item) for item in expected_keywords],
                    kg_knowledge=[str(item) for item in kg_knowledge],
                )
            )
    if not cases:
        raise ValueError(f"No eval cases loaded from {path}")
    return cases


def source_matches(source: Any, case: EvalCase) -> bool:
    haystack = "\n".join(
        [
            str(getattr(source, "source_type", "")),
            str(getattr(source, "source_title", "")),
            str(getattr(source, "section", "")),
            str(getattr(source, "content", "")),
        ]
    )
    if any(expected and expected in haystack for expected in case.expected_sources):
        return True
    if any(keyword and keyword in haystack for keyword in case.expected_keywords):
        return True
    return False


def hit_rank(sources: list[Any], case: EvalCase) -> int | None:
    for idx, source in enumerate(sources, start=1):
        if source_matches(source, case):
            return idx
    return None


def brief_sources(sources: list[Any], limit: int = 5) -> list[dict[str, Any]]:
    rows = []
    for source in sources[:limit]:
        rows.append(
            {
                "title": getattr(source, "source_title", ""),
                "section": getattr(source, "section", ""),
                "type": getattr(source, "source_type", ""),
                "score": round(
                    float(
                        getattr(source, "rerank_score", 0.0)
                        or getattr(source, "fused_score", 0.0)
                    ),
                    4,
                ),
            }
        )
    return rows


async def run_method(
    retriever: Any,
    session: Any,
    case: EvalCase,
    method: str,
    top_k: int,
    candidate_k: int,
) -> list[Any]:
    if method == "vector":
        embedder = get_embedding_service()
        embedding = await embedder.embed_query(case.query)
        return await retriever._vector_search(session, embedding, top_k)

    if method == "bm25":
        return await retriever._bm25_search(session, case.query, top_k)

    vector_embedding = await get_embedding_service().embed_query(case.query)
    vector_sources = await retriever._vector_search(session, vector_embedding, candidate_k)
    bm25_sources = await retriever._bm25_search(session, case.query, candidate_k)
    kg_sources = retriever._kg_sources(case.kg_knowledge)
    candidates = retriever._merge_candidates(
        [*vector_sources, *bm25_sources, *kg_sources], entities={}
    )
    candidates = sorted(candidates, key=lambda item: item.fused_score, reverse=True)
    selected_candidates = retriever._select_rerank_candidates(candidates, candidate_k)

    if method == "hybrid_fused":
        return candidates[:top_k]
    if method == "candidate_pool":
        return selected_candidates
    if method == "hybrid_rerank":
        ranked = await get_reranker_service().rerank(
            case.query, selected_candidates, top_k
        )
        for idx, source in enumerate(ranked, start=1):
            source.citation_id = idx
        return ranked
    raise ValueError(f"Unknown method: {method}")


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from sqlalchemy import func, select

    from app.db.models import DocumentChunk
    from app.db.session import async_session_maker, engine, init_db
    from app.services.embedding_service import get_embedding_service as _get_embedding
    from app.services.hybrid_retriever import get_hybrid_retriever
    from app.services.reranker_service import get_reranker_service as _get_reranker

    globals()["get_embedding_service"] = _get_embedding
    globals()["get_reranker_service"] = _get_reranker

    if args.init_db:
        await init_db()

    cases = load_eval_cases(args.eval_file)
    methods = ["vector", "bm25", "hybrid_fused", "hybrid_rerank", "candidate_pool"]
    stats = {method: MethodStats() for method in methods}
    failures: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    retriever = get_hybrid_retriever()

    try:
        async with async_session_maker() as session:
            chunk_count = await session.scalar(select(func.count()).select_from(DocumentChunk))
            if not chunk_count:
                print(
                    "WARNING: document_chunks is empty. Run document indexing before evaluation.",
                    file=sys.stderr,
                )

            for case in cases:
                for method in methods:
                    start = time.perf_counter()
                    sources = await run_method(
                        retriever,
                        session,
                        case,
                        method,
                        top_k=args.top_k,
                        candidate_k=args.candidate_k,
                    )
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    rank = hit_rank(sources, case)
                    stats[method].latency_ms.append(elapsed_ms)
                    if rank is not None:
                        stats[method].hits += 1
                        stats[method].reciprocal_rank_sum += 1 / rank
                    elif args.show_failures:
                        failures[method].append(
                            {
                                "id": case.case_id,
                                "query": case.query,
                                "expected_sources": case.expected_sources,
                                "expected_keywords": case.expected_keywords,
                                "top_sources": brief_sources(sources, args.top_k),
                            }
                        )

        total = len(cases)
        summary = {
            "eval_file": str(args.eval_file),
            "cases": total,
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "metrics": {},
        }
        for method, item in stats.items():
            latencies = item.latency_ms or [0.0]
            summary["metrics"][method] = {
                "recall": round(item.hits / total, 4),
                "mrr": round(item.reciprocal_rank_sum / total, 4),
                "hits": item.hits,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            }
        if args.show_failures:
            summary["failures"] = failures
        return summary
    finally:
        await engine.dispose()


def print_summary(summary: dict[str, Any]) -> None:
    print(
        f"eval_file={summary['eval_file']} cases={summary['cases']} "
        f"top_k={summary['top_k']} candidate_k={summary['candidate_k']}"
    )
    print("method            recall    mrr       hits      avg_latency_ms")
    for method, metrics in summary["metrics"].items():
        print(
            f"{method:<17} {metrics['recall']:<9.4f} "
            f"{metrics['mrr']:<9.4f} {metrics['hits']:<9} "
            f"{metrics['avg_latency_ms']:<.2f}"
        )
    if "failures" in summary:
        print("\nfailures:")
        print(json.dumps(summary["failures"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 2 P0 retrieval, fusion, and rerank metrics."
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "rag_eval" / "p0_retrieval_eval.jsonl",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--init-db", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--show-failures", action="store_true")
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
