from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class GenerationConfig:
    prompt: str
    negative_prompt: str
    output_dir: Path
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    seed: int = 42
    steps: int = 25
    guidance_scale: float = 7.0
    width: int = 768
    height: int = 768
    device: str = "auto"
    use_cpu_offload: bool = True


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(model_id: str, device: str, use_cpu_offload: bool):
    import torch
    from diffusers import AutoPipelineForText2Image

    dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs = {"torch_dtype": dtype}

    if device == "cuda":
        kwargs["variant"] = "fp16"
        kwargs["use_safetensors"] = True

    try:
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)
    except OSError:
        kwargs.pop("variant", None)
        kwargs["use_safetensors"] = False
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)

    if device == "cuda":
        if use_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
    else:
        pipe.to(device)

    return pipe

def generate_one(config: GenerationConfig, pipe=None) -> Path:
    import torch

    device = choose_device(config.device)
    if pipe is None:
        pipe = load_pipeline(config.model_id, device, config.use_cpu_offload)

    generator_device = "cuda" if device == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(config.seed)

    result = pipe(
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        num_inference_steps=config.steps,
        guidance_scale=config.guidance_scale,
        width=config.width,
        height=config.height,
        generator=generator,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = config.output_dir / f"sdxl_{timestamp}_seed{config.seed}.png"
    result.images[0].save(image_path)

    metadata_path = config.output_dir / "metadata.jsonl"
    record = {**asdict(config), "output_image": str(image_path.as_posix()), "device_used": device}
    record["output_dir"] = str(config.output_dir.as_posix())
    with metadata_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return image_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one SDXL fashion image from a text prompt.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("generated/week2"))
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-cpu-offload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GenerationConfig(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        output_dir=args.output_dir,
        model_id=args.model_id,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        device=args.device,
        use_cpu_offload=not args.no_cpu_offload,
    )
    image_path = generate_one(config)
    print(f"Saved image: {image_path}")


if __name__ == "__main__":
    main()

