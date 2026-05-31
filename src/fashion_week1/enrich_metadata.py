from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .schema import clean_list, normalize_text, read_jsonl, write_jsonl


def load_taxonomy(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def phrase_found(text: str, phrase: str) -> bool:
    phrase = normalize_text(phrase)
    if not phrase:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def extract_group(text: str, group: Dict[str, List[str]]) -> List[str]:
    matches: List[str] = []
    for canonical, synonyms in group.items():
        terms = [canonical] + list(synonyms)
        if any(phrase_found(text, term) for term in terms):
            matches.append(canonical)
    return clean_list(matches)


def infer_style_archetype(text: str, taxonomy: Dict[str, Any]) -> str:
    scores: Dict[str, int] = {}
    for style, terms in taxonomy.get("style_archetypes", {}).items():
        scores[style] = sum(1 for term in terms if phrase_found(text, term))
    if not scores:
        return ""
    best_style, best_score = max(scores.items(), key=lambda item: item[1])
    return best_style if best_score > 0 else ""


def infer_category(record: Dict[str, Any], taxonomy: Dict[str, Any], text: str) -> str:
    current = normalize_text(record.get("category", ""))
    if current and current != "unknown":
        return current
    for category in taxonomy.get("categories", []):
        if phrase_found(text, category):
            return category
    return current


def build_prompt(record: Dict[str, Any]) -> str:
    category = record.get("category") or "fashion garment"
    parts: List[str] = ["studio fashion product photograph", str(category)]
    for key in ["style_archetype", "colors", "materials", "patterns", "silhouettes", "details"]:
        value = record.get(key)
        if isinstance(value, list) and value:
            parts.append(", ".join(value))
        elif isinstance(value, str) and value:
            parts.append(value)
    parts.extend(["clean background", "high detail", "realistic fabric texture"])
    return ", ".join(clean_list(parts))


def build_retrieval_text(record: Dict[str, Any]) -> str:
    fields = [
        ("category", record.get("category")),
        ("subcategory", record.get("subcategory")),
        ("style", record.get("style_archetype")),
        ("colors", ", ".join(record.get("colors", []))),
        ("materials", ", ".join(record.get("materials", []))),
        ("patterns", ", ".join(record.get("patterns", []))),
        ("silhouettes", ", ".join(record.get("silhouettes", []))),
        ("occasions", ", ".join(record.get("occasions", []))),
        ("details", ", ".join(record.get("details", []))),
        ("caption", record.get("caption_normalized")),
    ]
    return " | ".join(f"{name}: {value}" for name, value in fields if value)


def enrich_record(record: Dict[str, Any], taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    text = normalize_text(
        " ".join(
            [
                str(record.get("caption_raw", "")),
                str(record.get("category", "")),
                str(record.get("subcategory", "")),
                str(record.get("brand", "")),
            ]
        )
    )
    groups = taxonomy.get("attribute_terms", {})
    record["caption_normalized"] = normalize_text(record.get("caption_raw", ""))
    record["category"] = infer_category(record, taxonomy, text)
    record["colors"] = extract_group(text, groups.get("colors", {}))
    record["materials"] = extract_group(text, groups.get("materials", {}))
    record["patterns"] = extract_group(text, groups.get("patterns", {}))
    record["silhouettes"] = extract_group(text, groups.get("silhouettes", {}))
    record["occasions"] = extract_group(text, groups.get("occasions", {}))
    record["details"] = extract_group(text, groups.get("details", {}))
    record["style_archetype"] = infer_style_archetype(text, taxonomy)
    record["design_prompt"] = build_prompt(record)
    record["negative_prompt"] = "low quality, blurry, distorted garment, unreadable logo, extra limbs, bad anatomy"
    record["retrieval_text"] = build_retrieval_text(record)
    return record


def enrich_manifest(input_path: Path, output_path: Path, taxonomy_path: Path) -> int:
    taxonomy = load_taxonomy(taxonomy_path)
    rows = [enrich_record(record, taxonomy) for record in read_jsonl(input_path)]
    write_jsonl(output_path, rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich fashion metadata with design attributes.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/manifest_raw.jsonl"))
    parser.add_argument("--taxonomy", type=Path, default=Path("config/taxonomy.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/manifest_enriched.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = enrich_manifest(args.input, args.output, args.taxonomy)
    print(f"Wrote {count} enriched records to {args.output}")


if __name__ == "__main__":
    main()

