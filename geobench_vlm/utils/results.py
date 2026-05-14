import json
from datetime import datetime, timezone
from pathlib import Path


def save_predictions(
    results: list[dict],
    results_dir: str,
    model_slug: str,
    mode: str,
    split_name: str,
) -> str:
    """Write results to results_dir/{model_slug}/{mode}/{split_name}.json."""
    out_dir = Path(results_dir) / model_slug / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{split_name}.json"
    txt_path = out_dir / f"{split_name}.txt"

    try:
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4, default=str)
        return str(json_path)
    except Exception as e:
        print(f"JSON serialisation failed ({e}), writing plain text fallback.")
        with open(txt_path, "w") as f:
            f.write(str(results))
        return str(txt_path)


def write_manifest(
    results_dir: str,
    model_slug: str,
    mode: str,
    data_path: str,
    split_name: str,
    score_summary: dict | None = None,
) -> str:
    """Write a manifest.json entry for this run, including timestamp and scores if provided."""
    out_dir = Path(results_dir) / model_slug / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    # Load existing manifest to append split entries
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {
            "model_slug": model_slug,
            "mode": mode,
            "runs": [],
        }

    manifest["runs"].append(
        {
            "split": split_name,
            "data_path": data_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scores": score_summary or {},
        }
    )

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    return str(manifest_path)
