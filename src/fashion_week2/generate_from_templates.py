from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

from .generate_sdxl import GenerationConfig, generate_one, load_pipeline, choose_device
from .prompt_library import render_prompt


def read_jsonl(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SDXL images from reusable fashion prompt templates.")
    parser.add_argument("--template-file", type=Path, default=Path("config/week2_prompt_templates.json"))
    parser.add_argument("--examples", type=Path, default=Path("examples/week2_prompt_examples.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated/week2"))
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    pipe = load_pipeline(args.model_id, device, use_cpu_offload=True)

    count = 0
    for row in read_jsonl(args.examples):
        template_name = row.pop("template", "studio_product")
        prompt_result = render_prompt(args.template_file, template_name, row)
        config = GenerationConfig(
            prompt=prompt_result.prompt,
            negative_prompt=prompt_result.negative_prompt,
            output_dir=args.output_dir,
            model_id=args.model_id,
            seed=42 + count,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            device=device,
        )
        image_path = generate_one(config, pipe=pipe)
        print(f"Saved {image_path}")
        count += 1
        if args.limit and count >= args.limit:
            break


if __name__ == "__main__":
    main()

