from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from .schema import normalize_text, stable_id


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

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

CATEGORY_ALIASES: Dict[str, str] = {
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
    return "clothing"


def clean_subcategory(folder_name: str) -> str:
    return folder_name.replace("_", " ").replace("-", " ").strip()


def build_caption(folder_name: str, category: str) -> str:
    subcategory = clean_subcategory(folder_name)
    return f"DeepFashion {category}: {subcategory}."


def image_rows(image_root: Path, raw_dir: Path, limit: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for image_path in sorted(image_root.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        folder_name = image_path.parent.name
        category = guess_category(folder_name)
        relative_path = image_path.resolve().relative_to(raw_dir.resolve()).as_posix()

        rows.append(
            {
                "image_path": relative_path,
                "item_id": f"deepfashion-{stable_id(relative_path)}",
                "source_dataset": "deepfashion_actual_images",
                "license": "non_commercial_research_only_deepfashion",
                "category": category,
                "subcategory": clean_subcategory(folder_name),
                "brand": "",
                "caption": build_caption(folder_name, category),
                "split": "",
            }
        )

        if limit and len(rows) >= limit:
            break

    return rows


def write_catalog(rows: List[Dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "item_id",
                "source_dataset",
                "license",
                "category",
                "subcategory",
                "brand",
                "caption",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a DeepFashion CSV from the image folders that exist locally.")
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/deepfashion_category/img"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/deepfashion_catalog.csv"))
    parser.add_argument("--limit", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = image_rows(args.image_root, args.raw_dir, args.limit)
    write_catalog(rows, args.output)
    print(f"Wrote {len(rows)} real image rows to {args.output}")
    if rows:
        print(f"First image path: {rows[0]['image_path']}")


if __name__ == "__main__":
    main()
