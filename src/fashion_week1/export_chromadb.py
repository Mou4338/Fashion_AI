from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List

from .schema import normalize_text, read_jsonl


def stable_hash_embedding(text: str, dims: int = 384) -> List[float]:
    """Offline placeholder embedding.

    Replace this with CLIP or FashionCLIP embeddings in Week 2. The goal in Week 1
    is to prove the retrieval plumbing without downloading a model.
    """
    vector = [0.0] * dims
    tokens = normalize_text(text).split()
    for token in tokens:
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dims
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def metadata_for_chroma(record: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {}
    for key in [
        "item_id",
        "image_path",
        "source_dataset",
        "license",
        "split",
        "category",
        "subcategory",
        "brand",
        "style_archetype",
        "quality_score",
    ]:
        value = record.get(key)
        if value is not None:
            allowed[key] = value
    for key in ["colors", "materials", "patterns", "silhouettes", "occasions", "details", "issue_flags"]:
        allowed[key] = ",".join(record.get(key, []))
    return allowed


def export_chroma(manifest: Path, persist_dir: Path, collection_name: str) -> int:
    try:
        import chromadb
    except ImportError as exc:
        raise SystemExit("Install chromadb first: pip install chromadb") from exc

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(collection_name)

    ids: List[str] = []
    documents: List[str] = []
    embeddings: List[List[float]] = []
    metadatas: List[Dict[str, Any]] = []

    for record in read_jsonl(manifest):
        text = record.get("retrieval_text") or record.get("caption_normalized") or record.get("caption_raw") or ""
        ids.append(record.get("record_id") or record.get("item_id"))
        documents.append(text)
        embeddings.append(stable_hash_embedding(text))
        metadatas.append(metadata_for_chroma(record))

    if ids:
        collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    return len(ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export curated fashion records to ChromaDB.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_splits.jsonl"))
    parser.add_argument("--persist-dir", type=Path, default=Path("data/processed/chroma"))
    parser.add_argument("--collection", default="fashion_week1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = export_chroma(args.manifest, args.persist_dir, args.collection)
    print(f"Indexed {count} records in ChromaDB collection '{args.collection}'.")


if __name__ == "__main__":
    main()

