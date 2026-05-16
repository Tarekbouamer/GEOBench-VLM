import os
from time import perf_counter

import torch
import typer
from torch.utils.data import DataLoader

from geobench_vlm.datasets.dataset import MultimodalDataset, collate_fn
from geobench_vlm.models.registry import TEMPORAL_MODELS
from geobench_vlm.utils.results import save_predictions, write_manifest
from geobench_vlm.utils.runner import run_eval
from geobench_vlm.utils.scoring import build_score_summary

app = typer.Typer(add_completion=False)


@app.command()
def main(
    model: str = typer.Option(
        ..., "--model", "-m", help=f"Model key. Choices: {', '.join(TEMPORAL_MODELS)}"
    ),
    data: str = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to dataset split directory (contains Temporal/qa.json).",
    ),
    results: str = typer.Option(
        ..., "--results", "-r", help="Directory to save prediction output."
    ),
    max_samples: int | None = typer.Option(
        None, "--max-samples", help="Limit number of questions (smoke test)."
    ),
    score: bool = typer.Option(
        False,
        "--score",
        "-s",
        is_flag=True,
        help="Compute and print scores after inference.",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        help="DataLoader batch size (default 1 for multi-image safety).",
    ),
):
    """Run temporal (pre/post image) evaluation for one model, save predictions, and optionally score."""
    if model not in TEMPORAL_MODELS:
        typer.echo(
            f"Unknown model '{model}'. Choose from: {', '.join(TEMPORAL_MODELS)}",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(
        f"[run_temporal] model={model}  data={data}  results={results}  max_samples={max_samples}"
    )

    # data
    dataset = MultimodalDataset(data, mode="temporal", max_samples=max_samples)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # model
    typer.echo(f"[run_temporal] Loading {model} ...")
    vlm = TEMPORAL_MODELS[model](mode="temporal")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # inference
    start_time = perf_counter()
    typer.echo("[run_temporal] Running inference ...")
    predictions = run_eval(vlm, loader, device)
    infer_time = perf_counter() - start_time

    # save
    split_name = os.path.basename(os.path.normpath(data))
    score_summary: dict = {}

    if score:
        score_summary = build_score_summary(predictions)
        typer.echo(
            f"\n[run_temporal] Overall accuracy: {score_summary.get('overall', 0):.2f}%"
        )
        for task, acc in sorted(score_summary.get("per_task", {}).items()):
            typer.echo(f"  {task}: {acc:.2f}%")

    pred_path = save_predictions(
        predictions, results, vlm.model_slug, "temporal", split_name
    )
    mfst_path = write_manifest(
        results,
        vlm.model_slug,
        "temporal",
        data,
        split_name,
        score_summary,
        batch_size=batch_size,
        sample_count=len(dataset),
        infer_time=infer_time,
        predictions_path=pred_path,
    )
    typer.echo(f"[run_temporal] Predictions → {pred_path}")
    typer.echo(f"[run_temporal] Manifest    → {mfst_path}")


if __name__ == "__main__":
    app()
