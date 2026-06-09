from __future__ import annotations

from pathlib import Path
from typing import Optional

import gradio as gr

from src.fashion_week2.generate_sdxl import GenerationConfig, generate_one, load_pipeline, choose_device
from src.fashion_week2.prompt_library import render_prompt


PIPE = None
DEVICE: Optional[str] = None


def generate(
    template: str,
    garment: str,
    color: str,
    material: str,
    silhouette: str,
    style: str,
    details: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
):
    global PIPE, DEVICE
    template_file = Path("config/week2_prompt_templates.json")
    prompt_result = render_prompt(
        template_file,
        template,
        {
            "garment": garment,
            "color": color,
            "material": material,
            "silhouette": silhouette,
            "style": style,
            "details": details,
        },
    )

    if PIPE is None:
        DEVICE = choose_device("auto")
        PIPE = load_pipeline("stabilityai/stable-diffusion-xl-base-1.0", DEVICE, use_cpu_offload=True)

    config = GenerationConfig(
        prompt=prompt_result.prompt,
        negative_prompt=prompt_result.negative_prompt,
        output_dir=Path("generated/week2"),
        seed=seed,
        steps=steps,
        width=width,
        height=height,
        device=DEVICE or "auto",
    )
    image_path = generate_one(config, pipe=PIPE)
    return str(image_path), prompt_result.prompt


with gr.Blocks(title="Week 2 SDXL Fashion Studio") as demo:
    gr.Markdown("# Week 2 SDXL Fashion Studio")
    with gr.Row():
        template = gr.Dropdown(["studio_product", "runway_editorial", "flat_lay", "lookbook"], value="studio_product")
        seed = gr.Number(value=42, precision=0, label="Seed")
    with gr.Row():
        garment = gr.Textbox(value="midi dress", label="Garment")
        color = gr.Textbox(value="black", label="Color")
        material = gr.Textbox(value="satin", label="Material")
    with gr.Row():
        silhouette = gr.Textbox(value="fitted", label="Silhouette")
        style = gr.Textbox(value="evening", label="Style")
        details = gr.Textbox(value="square neckline, soft pleats", label="Details")
    with gr.Row():
        steps = gr.Slider(10, 50, value=25, step=1, label="Steps")
        width = gr.Dropdown([512, 768, 1024], value=768, label="Width")
        height = gr.Dropdown([512, 768, 1024], value=768, label="Height")
    button = gr.Button("Generate")
    image = gr.Image(label="Generated Design")
    prompt = gr.Textbox(label="Final Prompt", lines=4)
    button.click(
        generate,
        inputs=[template, garment, color, material, silhouette, style, details, seed, steps, width, height],
        outputs=[image, prompt],
    )


if __name__ == "__main__":
    demo.launch()

