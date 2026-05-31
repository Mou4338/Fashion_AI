from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

BASE_RECORD: Dict[str, Any] = {
    "record_id": "",
    "item_id": "",
    "image_path": "",
    "source_dataset": "",
    "license": "",
    "split": "",
    "category": "",
    "subcategory": "",
    "brand": "",
    "season": "",
    "gender": "",
    "caption_raw": "",
    "caption_normalized": "",
    "colors": [],
    "materials": [],
    "patterns": [],
    "silhouettes": [],
    "occasions": [],
    "details": [],
    "style_archetype": "",
    "design_prompt": "",
    "negative_prompt": "",
    "retrieval_text": "",
    "width": None,
    "height": None,
    "sha256": "",
    "quality_score": 0,
    "issue_flags": [],
}


def normalize_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\-\s/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def slugify(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "unknown"


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*parts: str, length: int = 16) -> str:
    raw = "::".join(str(part or "") for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def clean_list(values: Iterable[str]) -> List[str]:
    seen = set()
    cleaned: List[str] = []
    for value in values:
        value = normalize_text(value)
        if value and value not in seen:
            seen.add(value)
            cleaned.append(value)
    return cleaned


def base_record(**updates: Any) -> Dict[str, Any]:
    record = dict(BASE_RECORD)
    for key, value in updates.items():
        if key in record:
            record[key] = value
    return record

