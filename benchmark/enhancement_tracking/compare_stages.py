"""Compare two enhancement-stage artifacts using paired deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") != "enhancement-stage-v1":
        raise ValueError(f"Unsupported stage artifact: {path}")
    return artifact


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def _format_delta(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "n/a"
    return f"{100 * (after - before):+.2f} pp"


def _metric_row(label: str, before: float | None, after: float | None) -> str:
    return (
        f"| {label} | {_format_rate(before)} | {_format_rate(after)} | "
        f"{_format_delta(before, after)} |"
    )


def _paired_decision_changes(before: dict, after: dict) -> tuple[list[str], list[str]]:
    before_rows = {
        (row["dialogue_id"], row["turn_id"]): row
        for row in before["multiturn"]["control"]["rows"]
    }
    after_rows = {
        (row["dialogue_id"], row["turn_id"]): row
        for row in after["multiturn"]["control"]["rows"]
    }
    improved = []
    regressed = []
    for key in sorted(before_rows.keys() & after_rows.keys()):
        was_correct = before_rows[key]["decision_correct"]
        is_correct = after_rows[key]["decision_correct"]
        label = "/".join(key)
        if not was_correct and is_correct:
            improved.append(label)
        elif was_correct and not is_correct:
            regressed.append(label)
    return improved, regressed


def render(before: dict, after: dict) -> str:
    for dataset_name in ("multiturn", "propagation"):
        before_hash = before["datasets"][dataset_name]["sha256"]
        after_hash = after["datasets"][dataset_name]["sha256"]
        if before_hash != after_hash:
            raise ValueError(
                f"Cannot make a paired comparison: {dataset_name} dataset hashes differ"
            )

    before_control = before["multiturn"]["control"]["summary"]
    after_control = after["multiturn"]["control"]["summary"]
    before_proxy = before["propagation_proxy"]["summary"]
    after_proxy = after["propagation_proxy"]["summary"]
    lines = [
        f"# {before['stage']['id']} → {after['stage']['id']}",
        "",
        "| Metric | Before | After | Delta |",
        "|---|---:|---:|---:|",
    ]
    for tier in ("core", "challenge"):
        lines.append(
            _metric_row(
                f"Multi-turn {tier} decision accuracy",
                before_control[tier]["decision_accuracy"],
                after_control[tier]["decision_accuracy"],
            )
        )
        lines.append(
            _metric_row(
                f"Multi-turn {tier} resolved-query accuracy",
                before_control[tier]["resolved_query_accuracy"],
                after_control[tier]["resolved_query_accuracy"],
            )
        )
    lines.extend(
        [
            _metric_row(
                "Frozen propagation answer-span rate",
                before_proxy["answer_span_rate_after"],
                after_proxy["answer_span_rate_after"],
            ),
            _metric_row(
                "Frozen propagation corpus token error rate (lower is better)",
                before_proxy["corpus_token_error_rate_after"],
                after_proxy["corpus_token_error_rate_after"],
            ),
        ]
    )

    before_llm = before["multiturn"].get("llm")
    after_llm = after["multiturn"].get("llm")
    if before_llm and after_llm:
        lines.append(
            _metric_row(
                "Multi-turn core LLM keyword accuracy",
                before_llm["summary"]["core"]["llm_keyword_accuracy"],
                after_llm["summary"]["core"]["llm_keyword_accuracy"],
            )
        )

    improved, regressed = _paired_decision_changes(before, after)
    lines.extend(
        [
            "",
            "Paired control-turn changes:",
            "",
            f"- Improved ({len(improved)}): {', '.join(improved) if improved else 'none'}",
            f"- Regressed ({len(regressed)}): {', '.join(regressed) if regressed else 'none'}",
            "",
            (
                "The propagation answer-span and token-error rows are lexical proxy "
                "metrics, not full downstream LLM QA results."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rendered = render(_load(args.before), _load(args.after))
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
