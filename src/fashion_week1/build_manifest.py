from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .schema import IMAGE_EXTENSIONS, base_record, file_sha256, normalize_text, slugify, stable_id, write_jsonl


try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None


CATEGORY_WORDS = [
    "dress",
    "blouse",
    "tank",
    "tee",
    "top",
    "shirt",
    "sweater",
    "skirt",
    "jacket",
    "coat",
    "shorts",
    "pants",
    "jeans",
    "romper",
    "jumpsuit",
    "cardigan",
    "blazer",
    "hoodie",
    "leggings",
    "kimono",
    "poncho",
]

CATEGORY_ALIASES = {
    "tee": "t-shirt",
    "tank": "top",
    "leggings": "pants",
}


def guess_category(folder_name: str) -> str:
    text = normalize_text(folder_name.replace("_", " ").replace("-", " "))
    words = text.split()
    for word in reversed(words):
        if word in CATEGORY_WORDS:
            return CATEGORY_ALIASES.get(word, word)
    return slugify(folder_name) if folder_name else "clothing"


def clean_name(name: str) -> str:
    return str(name or "").replace("_", " ").replace("-", " ").strip()


def image_size(path: Path) -> tuple[Optional[int], Optional[int]]:
    if Image is None or not path.exists():
        return None, None
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None


def resolve_image_path(raw_dir: Path, value: str) -> Path:
    text = str(value or "").strip().strip('"').replace("\\", "/")
    text = text.replace("deepfashion_category/img/img/", "deepfashion_category/img/")

    direct = Path(text)
    candidates: List[Path] = []

    if direct.is_absolute():
        candidates.append(direct)
    else:
        candidates.append(raw_dir / text)
        candidates.append(Path(text))

        if text.startswith("data/raw/"):
            candidates.append(raw_dir / text.removeprefix("data/raw/"))

        if text.startswith("img/"):
            candidates.append(raw_dir / "deepfashion_category" / text)

        if not text.startswith("deepfashion_category/"):
            candidates.append(raw_dir / "deepfashion_category" / "img" / text)

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate

    return candidates[0]


def row_to_record(raw_dir: Path, row: Dict[str, str], default_source: str) -> Dict[str, Any]:
    image_path = resolve_image_path(raw_dir, row.get("image_path", ""))
    width, height = image_size(image_path)
    sha = file_sha256(image_path) if image_path.exists() else ""
    caption = row.get("caption", "") or row.get("description", "") or row.get("caption_raw", "")
    category = row.get("category", "") or guess_category(image_path.parent.name)
    subcategory = row.get("subcategory", "") or clean_name(image_path.parent.name)
    source = row.get("source_dataset", "") or default_source
    item_id = row.get("item_id", "") or stable_id(source, category, image_path.stem)

    flags: List[str] = []
    if not image_path.exists():
        flags.append("missing_image")
    if width is None or height is None:
        flags.append("missing_dimensions")

    return base_record(
        record_id=stable_id(source, item_id, str(image_path)),
        item_id=item_id,
        image_path=str(image_path.as_posix()),
        source_dataset=source,
        license=row.get("license", "") or "research_only_check_source_terms",
        split=row.get("split", ""),
        category=slugify(category),
        subcategory=subcategory,
        brand=row.get("brand", ""),
        season=row.get("season", ""),
        gender=row.get("gender", ""),
        caption_raw=caption,
        width=width,
        height=height,
        sha256=sha,
        issue_flags=flags,
    )


def records_from_catalog(raw_dir: Path, catalog: Path, default_source: str) -> Iterable[Dict[str, Any]]:
    with catalog.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row_to_record(raw_dir, row, default_source)


def records_from_folders(raw_dir: Path, default_source: str) -> Iterable[Dict[str, Any]]:
    for image_path in sorted(raw_dir.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        width, height = image_size(image_path)
        sha = file_sha256(image_path) if image_path.exists() else ""
        relative = image_path.relative_to(raw_dir)
        source = relative.parts[0] if len(relative.parts) > 1 else default_source
        category = guess_category(image_path.parent.name)
        subcategory = clean_name(image_path.parent.name)
        item_id = stable_id(source, subcategory, image_path.stem)
        yield base_record(
            record_id=stable_id(source, item_id, str(image_path)),
            item_id=item_id,
            image_path=str(image_path.as_posix()),
            source_dataset=source,
            license="research_only_check_source_terms",
            category=slugify(category),
            subcategory=subcategory,
            caption_raw=f"DeepFashion {category}: {subcategory}.",
            width=width,
            height=height,
            sha256=sha,
        )


def build_manifest(raw_dir: Path, output: Path, catalog: Optional[Path], default_source: str) -> int:
    if catalog:
        rows = list(records_from_catalog(raw_dir, catalog, default_source))
    else:
        rows = list(records_from_folders(raw_dir, default_source))
    write_jsonl(output, rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a raw fashion dataset manifest.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/processed/manifest_raw.jsonl"))
    parser.add_argument("--source", default="deepfashion_category")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = build_manifest(args.raw_dir, args.output, args.catalog, args.source)
    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
