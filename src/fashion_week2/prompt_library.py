from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass
class PromptResult:
    template_name: str
    prompt: str
    negative_prompt: str
    values: Dict[str, str]


def load_template_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def render_prompt(template_file: Path, template_name: str, values: Dict[str, str]) -> PromptResult:
    data = load_template_file(template_file)
    templates = data.get("templates", {})
    defaults = data.get("defaults", {})
    if template_name not in templates:
        available = ", ".join(sorted(templates))
        raise KeyError(f"Unknown template '{template_name}'. Available templates: {available}")

    merged = {**defaults, **{key: value for key, value in values.items() if value}}
    prompt = templates[template_name].format_map(SafeFormatDict(merged))
    negative_prompt = data.get("negative_prompt", "")
    return PromptResult(template_name=template_name, prompt=prompt, negative_prompt=negative_prompt, values=merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a reusable fashion prompt template.")
    parser.add_argument("--template-file", type=Path, default=Path("config/week2_prompt_templates.json"))
    parser.add_argument("--template", default="studio_product")
    parser.add_argument("--garment", default="")
    parser.add_argument("--color", default="")
    parser.add_argument("--material", default="")
    parser.add_argument("--silhouette", default="")
    parser.add_argument("--style", default="")
    parser.add_argument("--details", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = render_prompt(
        args.template_file,
        args.template,
        {
            "garment": args.garment,
            "color": args.color,
            "material": args.material,
            "silhouette": args.silhouette,
            "style": args.style,
            "details": args.details,
        },
    )
    print("PROMPT:")
    print(result.prompt)
    print()
    print("NEGATIVE PROMPT:")
    print(result.negative_prompt)


if __name__ == "__main__":
    main()

