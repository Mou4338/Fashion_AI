from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


def read_metadata(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return pil_to_tensor(image)


def evaluate(metadata_file: Path, output: Path, model_name: str) -> Dict[str, object]:
    from torchmetrics.multimodal.clip_score import CLIPScore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = CLIPScore(model_name_or_path=model_name).to(device)
    rows: List[Dict[str, object]] = []

    for record in read_metadata(metadata_file):
        image_path = Path(record["output_image"])
        prompt = record["prompt"]
        image = load_image(image_path).unsqueeze(0).to(device)
        score = metric(image, [prompt]).detach().cpu().item()
        rows.append({"image_path": str(image_path.as_posix()), "prompt": prompt, "clip_score": round(score, 4)})

    average = sum(row["clip_score"] for row in rows) / max(len(rows), 1)
    report = {"count": len(rows), "average_clip_score": round(average, 4), "items": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated image/text alignment with CLIPScore.")
    parser.add_argument("--metadata-file", type=Path, default=Path("generated/week2/metadata.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/week2_clip_scores.json"))
    parser.add_argument("--model-name", default="openai/clip-vit-large-patch14")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(args.metadata_file, args.output, args.model_name)
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()

