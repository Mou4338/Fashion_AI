from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .schema import read_jsonl, write_jsonl


def score_record(record: Dict[str, Any], duplicate_hashes: set[str], min_size: int) -> Dict[str, Any]:
    flags = set(record.get("issue_flags", []))
    score = 0

    image_path = Path(record.get("image_path", ""))
    if image_path.exists():
        score += 20
    else:
        flags.add("missing_image")

    width = record.get("width") or 0
    height = record.get("height") or 0
    shortest_side = min(width, height) if width and height else 0

    if shortest_side >= min_size:
        score += 15
    elif width and height:
        score += 8
        flags.add("low_resolution")
    else:
        flags.add("missing_dimensions")

    if width and height:
        aspect = width / max(height, 1)
        if 0.35 <= aspect <= 2.8:
            score += 10
        else:
            flags.add("unusual_aspect_ratio")

    caption_words = str(record.get("caption_normalized", "") or record.get("caption_raw", "")).split()
    if len(caption_words) >= 8:
        score += 15
    elif len(caption_words) >= 4:
        score += 8
    else:
        flags.add("weak_caption")

    if record.get("category") and record.get("category") not in {"unknown", "clothing"}:
        score += 10
    else:
        flags.add("missing_category")

    for key in ["colors", "materials", "patterns", "silhouettes", "details"]:
        if record.get(key):
            score += 5

    sha = record.get("sha256", "")
    if sha and sha in duplicate_hashes:
        flags.add("duplicate_image")
    else:
        score += 10

    record["quality_score"] = min(score, 100)
    record["issue_flags"] = sorted(flags)
    return record


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    categories = Counter(row.get("category") or "unknown" for row in rows)
    flags = Counter(flag for row in rows for flag in row.get("issue_flags", []))
    scores = [row.get("quality_score", 0) for row in rows]
    return {
        "record_count": len(rows),
        "average_quality_score": round(sum(scores) / max(len(scores), 1), 2),
        "min_quality_score": min(scores) if scores else 0,
        "max_quality_score": max(scores) if scores else 0,
        "category_counts": dict(categories.most_common()),
        "issue_counts": dict(flags.most_common()),
    }


def write_issues(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["record_id", "image_path", "category", "quality_score", "issue_flags"],
        )
        writer.writeheader()
        for row in rows:
            if row.get("issue_flags"):
                writer.writerow(
                    {
                        "record_id": row.get("record_id", ""),
                        "image_path": row.get("image_path", ""),
                        "category": row.get("category", ""),
                        "quality_score": row.get("quality_score", 0),
                        "issue_flags": ";".join(row.get("issue_flags", [])),
                    }
                )


def run_quality(input_path: Path, output_path: Path, report_path: Path, issues_path: Path, min_size: int) -> Dict[str, Any]:
    rows = list(read_jsonl(input_path))
    hash_counts = Counter(row.get("sha256", "") for row in rows if row.get("sha256"))
    duplicate_hashes = {sha for sha, count in hash_counts.items() if count > 1}
    scored = [score_record(row, duplicate_hashes, min_size) for row in rows]
    report = summarize(scored)
    write_jsonl(output_path, scored)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_issues(issues_path, scored)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score curated fashion dataset quality.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/manifest_enriched.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/manifest_curated.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("reports/week1_quality_report.json"))
    parser.add_argument("--issues", type=Path, default=Path("reports/week1_issues.csv"))
    parser.add_argument("--min-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_quality(args.input, args.output, args.report, args.issues, args.min_size)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
