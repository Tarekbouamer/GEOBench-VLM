import os

import torch
import typer
from rich.console import Console
from rich.panel import Panel
from torch.utils.data import DataLoader

from geobench_vlm.datasets import MultimodalDataset, collate_fn
from geobench_vlm.models import SINGLE_MODELS
from geobench_vlm.utils import (
    build_score_summary,
    run_eval,
    save_predictions,
    write_manifest,
)

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    model: str = typer.Option(
        ..., "--model", "-m", help=f"Model key. Choices: {', '.join(SINGLE_MODELS)}"
    ),
    data: str = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to dataset split directory (contains Single/qa.json).",
    ),
    results: str = typer.Option(
        ..., "--results", "-r", help="Directory to save prediction output."
    ),
    max_samples: int | None = typer.Option(
        None, "--max-samples", help="Limit number of questions (smoke test)."
    ),
    batch_size: int = typer.Option(32, "--batch-size", help="DataLoader batch size."),
):
    """Run single-image evaluation for one model, save predictions, and optionally score."""
    if model not in SINGLE_MODELS:
        console.print(
            f"[red]Unknown model '{model}'. Choose from: {', '.join(SINGLE_MODELS)}[/red]"
        )
        raise typer.Exit(code=1)

    run_lines = [
        f"[bold]Model:[/bold] {model}",
        f"[bold]Data:[/bold] {data}",
        f"[bold]Results:[/bold] {results}",
        f"[bold]Max samples:[/bold] {max_samples if max_samples is not None else 'all'}",
        f"[bold]Batch size:[/bold] {batch_size}",
    ]
    console.print(
        Panel("\n".join(run_lines), title="[bold blue]Run Configuration[/bold blue]")
    )

    # data
    dataset = MultimodalDataset(data, mode="single", max_samples=max_samples)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # model
    with console.status(f"[bold cyan]Loading {model}...[/bold cyan]"):
        vlm = SINGLE_MODELS[model](mode="single")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # inference
    console.print("[bold cyan]Running inference...[/bold cyan]")
    predictions = run_eval(vlm, loader, device)

    # save
    split_name = os.path.basename(os.path.normpath(data))
    score_summary = build_score_summary(predictions)
    console.print()
    score_lines = [
        f"[bold]Model:[/bold] {vlm.model_slug}",
        f"[bold]Overall accuracy:[/bold] {score_summary.get('overall', 0):.2f}%",
    ]
    for task, acc in sorted(score_summary.get("per_task", {}).items()):
        score_lines.append(f"[bold]{task}:[/bold] {acc:.2f}%")

    console.print(
        Panel("\n".join(score_lines), title="[bold green]Scores[/bold green]")
    )

    pred_path = save_predictions(
        predictions, results, vlm.model_slug, "single", split_name
    )
    mfst_path = write_manifest(
        results, vlm.model_slug, "single", data, split_name, score_summary
    )
    output_lines = [
        f"[bold]Predictions:[/bold] {pred_path}",
        f"[bold]Manifest:[/bold] {mfst_path}",
    ]
    console.print(
        Panel("\n".join(output_lines), title="[bold magenta]Outputs[/bold magenta]")
    )


if __name__ == "__main__":
    app()
