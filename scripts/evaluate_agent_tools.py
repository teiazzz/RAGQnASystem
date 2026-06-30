"""Evaluate Agent tool-selection accuracy on a small medical routing set.

Default mode uses the deterministic fallback router so it is reproducible and
does not call an external LLM. Add ``--use-llm`` to evaluate the real planner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services import agent_tools

DEFAULT_CASES = Path("data/agent_eval/tool_use_eval_cases.jsonl")


def load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cases.append(json.loads(line))
    return cases


async def evaluate_case(case: dict, use_llm: bool) -> dict:
    calls, planner, planner_error = await agent_tools.select_tool_calls(
        case["query"],
        case.get("entities") or {},
        use_llm=use_llm,
    )
    predicted = [call.name for call in calls]
    expected = list(case.get("expected_tools") or [])
    predicted_set = set(predicted)
    expected_set = set(expected)
    overlap = predicted_set & expected_set
    precision = len(overlap) / len(predicted_set) if predicted_set else (1.0 if not expected else 0.0)
    recall = len(overlap) / len(expected_set) if expected_set else (1.0 if not predicted else 0.0)
    first_tool_ok = (
        True
        if not expected
        else bool(predicted) and predicted[0] == expected[0]
    )
    return {
        "id": case["id"],
        "query": case["query"],
        "expected": expected,
        "predicted": predicted,
        "planner": planner,
        "planner_error": planner_error,
        "exact_match": predicted_set == expected_set,
        "first_tool_ok": first_tool_ok,
        "precision": precision,
        "recall": recall,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--use-llm", action="store_true", help="call the configured LLM planner")
    parser.add_argument("--show-failures", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    results = [await evaluate_case(case, args.use_llm) for case in cases]
    summary = {
        "cases": len(results),
        "planner": "llm" if args.use_llm else "heuristic",
        "exact_match": round(mean(item["exact_match"] for item in results), 4),
        "first_tool_accuracy": round(mean(item["first_tool_ok"] for item in results), 4),
        "tool_precision": round(mean(item["precision"] for item in results), 4),
        "tool_recall": round(mean(item["recall"] for item in results), 4),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.show_failures:
        failures = [item for item in results if not item["exact_match"]]
        if failures:
            print("\nFailures:")
            for item in failures:
                print(
                    json.dumps(
                        {
                            "id": item["id"],
                            "expected": item["expected"],
                            "predicted": item["predicted"],
                            "query": item["query"],
                            "planner_error": item["planner_error"],
                        },
                        ensure_ascii=False,
                    )
                )


if __name__ == "__main__":
    asyncio.run(main())
