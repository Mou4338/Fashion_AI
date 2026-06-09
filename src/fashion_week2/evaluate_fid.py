from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_files(folder: Path, limit: int) -> Iterable[Path]:
    count = 0
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        yield path
        count += 1
        if limit and count >= limit:
            break


def load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return pil_to_tensor(image)


def update_metric(metric, folder: Path, real: bool, limit: int, device: str) -> int:
    count = 0
    for path in image_files(folder, limit):
        tensor = load_image(path).unsqueeze(0).to(device)
        metric.update(tensor, real=real)
        count += 1
    return count


def evaluate_fid(real_dir: Path, fake_dir: Path, output: Path, limit: int, feature: int) -> dict:
    from torchmetrics.image.fid import FrechetInceptionDistance

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = FrechetInceptionDistance(feature=feature, normalize=False).to(device)
    real_count = update_metric(metric, real_dir, real=True, limit=limit, device=device)
    fake_count = update_metric(metric, fake_dir, real=False, limit=limit, device=device)
    fid_score = metric.compute().detach().cpu().item()
    report = {"real_count": real_count, "fake_count": fake_count, "fid": round(fid_score, 4), "feature": feature}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate FID between real DeepFashion images and generated images.")
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--fake-dir", type=Path, default=Path("generated/week2"))
    parser.add_argument("--output", type=Path, default=Path("reports/week2_fid.json"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--feature", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_fid(args.real_dir, args.fake_dir, args.output, args.limit, args.feature)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

