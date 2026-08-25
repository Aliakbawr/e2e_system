#!/usr/bin/env python3
"""Build the deterministic integrity manifest for the research layout."""

from __future__ import annotations

import hashlib
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_DIR / "analysis/manifests/current/RESEARCH_ARTIFACTS_V1.sha256"

INCLUDE_FILES = {
    PROJECT_DIR / ".gitignore",
    PROJECT_DIR / "README.md",
    PROJECT_DIR / "analysis/README.md",
    PROJECT_DIR / "analysis/manifests/PATH_RELOCATION_V1.md",
    PROJECT_DIR / "analyze_answer_span_preservation.py",
    PROJECT_DIR / "prepare_downstream_eval.py",
    PROJECT_DIR / "prepare_vosk_downstream_eval.py",
    PROJECT_DIR / "data/test.tsv",
    PROJECT_DIR / "data/test.tar.gz",
    PROJECT_DIR / "qa/fleurs_fa_ir_generated_qa_candidates_v1.csv",
}

INCLUDE_TREES = [
    PROJECT_DIR / "asr",
    PROJECT_DIR / "llm",
    PROJECT_DIR / "analysis/inputs",
    PROJECT_DIR / "analysis/results",
    PROJECT_DIR / "analysis/reports",
    PROJECT_DIR / "analysis/manifests/historical_pre_restructure",
    PROJECT_DIR / "notebooks",
    PROJECT_DIR / "qa",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_files() -> list[Path]:
    files = {path for path in INCLUDE_FILES if path.is_file()}
    for directory in INCLUDE_TREES:
        files.update(
            path
            for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    files.add(Path(__file__).resolve())
    files.discard(OUTPUT_PATH)
    return sorted(files, key=lambda path: path.relative_to(PROJECT_DIR).as_posix())


def main() -> int:
    files = selected_files()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{sha256_file(path)}  {path.relative_to(PROJECT_DIR).as_posix()}"
        for path in files
    ]
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Manifest entries: {len(lines)}")
    print(f"Saved: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
