import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

from geobench_vlm.models import SINGLE_MODELS
from geobench_vlm.utils import build_mcq_prompt

app = typer.Typer(add_completion=False)
console = Console()


def resolve_image(data_path: str, image_id: str | None) -> tuple[str, dict]:
    """Resolve an image path and its QA entry from Single/qa.json."""
    data_path = Path(data_path)
    single_root = data_path / "Single"

    # qa annotations
    qa_path = single_root / "qa.json"

    assert (
        qa_path.is_file()
    ), f"No qa.json found at {qa_path}. Expected layout: {data_path}/Single/qa.json"

    with open(qa_path) as f:
        qa_entries: list[dict] = json.load(f)

    # If image_id is provided
    if image_id:
        for entry in qa_entries:
            if entry.get("image_name") != image_id:
                continue

            image_path = data_path / entry["image_path"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Image '{image_id}' not found at {image_path}")
            return str(image_path), entry

        raise FileNotFoundError(f"Image '{image_id}' not found in {data_path}")

    entry = qa_entries[0]
    image_path = data_path / entry["image_path"]
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found at {image_path}")

    return str(image_path), entry


@app.command()
def main(
    model: str = typer.Option(
        ..., "--model", "-m", help=f"Model key. Choices: {', '.join(SINGLE_MODELS)}"
    ),
    data: str = typer.Option(
        ..., "--data", help="Dataset root folder (contains Single/qa.json)."
    ),
    image: str = typer.Option("", "--image", "-i", help="Image filename or path."),
) -> None:
    """Inspect one image from Single/qa.json and compare model output to ground truth."""

    if model not in SINGLE_MODELS:
        console.print(
            f"[red]Unknown model '{model}'. Available: {', '.join(SINGLE_MODELS)}[/red]"
        )
        raise typer.Exit(code=1)

    # get image and QA entry
    img_path, qa = resolve_image(data, image.strip() or None)

    console.print(f"[bold cyan]Inspecting {img_path} with {model}...[/bold cyan]")

    question = qa["prompts"][0]
    options = qa.get("options", "")
    cls_desc = qa.get("cls_description", "")
    task = qa.get("task", "")
    ground_truth = str(
        qa.get("ground_truth_option", qa.get("ground_truth", ""))
    ).strip()

    # Build MCQ prompt
    qtype = "Multiple Choice Questions" if options else "Open Ended"
    prompt = build_mcq_prompt(question, options, cls_desc) if options else question

    # Show input panel before inference
    console.print()
    input_lines = [
        f"[bold]Image:[/bold]  {img_path}",
        f"[bold]Task:[/bold]   {task or '—'}",
        f"[bold]Question type:[/bold] {qtype}",
        f"[bold]Prompt:[/bold]\n{prompt}",
    ]
    console.print(
        Panel("\n".join(input_lines), title="[bold blue]Image + Prompt[/bold blue]")
    )

    # load model
    for _ in tqdm(
        range(1),
        desc=f"Loading {model}",
        unit="step",
        dynamic_ncols=True,
    ):
        vlm = SINGLE_MODELS[model](mode="single")

    # inference
    predicted = None
    for _ in tqdm(
        range(1),
        desc=f"{vlm.model_slug} infer",
        unit="step",
        dynamic_ncols=True,
    ):
        predicted = vlm.generate_response(
            img_path,
            prompt,
            qtype,
            options or None,
            cls_desc or None,
        )

    truth = ground_truth.upper()
    is_match = bool(truth and predicted and predicted == truth)

    console.print()
    output_lines = [
        f"[bold]Model:[/bold]  {vlm.model_slug}",
        f"[bold]Predicted :[/bold] {predicted or '—'}",
        f"[bold]Ground truth :[/bold] {truth or '—'}",
        f"[bold]Match:[/bold] {('✓ yes' if is_match else '✗ no') if truth else '—'}",
    ]

    console.print(
        Panel("\n".join(output_lines), title="[bold green]Answer[/bold green]")
    )

    console.print()


if __name__ == "__main__":
    app()
