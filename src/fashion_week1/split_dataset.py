from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict

from .schema import read_jsonl, write_jsonl


def bucket(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def choose_split(item_id: str, train: float, val: float, test: float) -> str:
    total = train + val + test
    if abs(total - 1.0) > 0.001:
        raise ValueError("train + val + test must equal 1.0")
    value = bucket(item_id)
    if value < train:
        return "train"
    if value < train + val:
        return "val"
    return "test"


def split_manifest(input_path: Path, output_path: Path, train: float, val: float, test: float) -> Dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    rows = []
    for record in read_jsonl(input_path):
        item_id = record.get("item_id") or record.get("record_id") or record.get("image_path")
        split = choose_split(str(item_id), train, val, test)
        record["split"] = split
        counts[split] += 1
        rows.append(record)
    write_jsonl(output_path, rows)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic dataset splits.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/manifest_curated.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/manifest_splits.jsonl"))
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = split_manifest(args.input, args.output, args.train, args.val, args.test)
    print(counts)


if __name__ == "__main__":
    main()

