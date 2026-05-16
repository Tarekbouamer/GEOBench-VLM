import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import torch

from geobench_vlm import __version__

from .runtime import infer_attn_impl


def is_colab_runtime() -> bool:
    return bool(os.getenv("COLAB_RELEASE_TAG") or os.getenv("COLAB_GPU"))


def safe_package_version(package_name: str) -> str | None:
    """Return installed package version."""
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


# Git info retrieval adapted
def get_git_info() -> tuple[str | None, bool | None]:
    """Return git commit SHA and dirty status."""

    # Get the repository root directory
    repo_root = Path(__file__).resolve().parents[2]
    env_commit = (
        os.getenv("GIT_COMMIT") or os.getenv("COMMIT_SHA") or os.getenv("GITHUB_SHA")
    )

    try:
        # Get the current commit hash
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )

        # Check for uncommitted changes
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return commit, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        #
        return env_commit, None


def build_runtime_metadata() -> dict:
    """Collect runtime metadata for the current environment."""
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "runtime": "colab" if is_colab_runtime() else "local",
        "torch": safe_package_version("torch"),
        "transformers": safe_package_version("transformers"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "gpu_memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1e9,
            2,
        )
        if cuda_available
        else None,
        "attention_backend": infer_attn_impl(),
        "geobench_vlm": __version__,
    }


def build_manifest_metadata(
    *,
    batch_size: int | None,
    sample_count: int | None,
    infer_time: float | None,
    predictions_path: str,
) -> dict:

    # Git
    git_commit, git_dirty = get_git_info()

    return {
        "batch_size": batch_size,
        "sample_count": sample_count,
        "infer_time": infer_time,
        "predictions_path": predictions_path,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        **build_runtime_metadata(),
    }


def save_predictions(
    results: list[dict],
    results_dir: str,
    model_slug: str,
    mode: str,
    split_name: str,
) -> str:
    """Save prediction outputs."""
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
    batch_size: int | None = None,
    sample_count: int | None = None,
    infer_time: float | None = None,
    predictions_path: str = "",
) -> str:
    """Write manifest.json for a benchmark run."""
    out_dir = Path(results_dir) / model_slug / mode

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    try:
        runtime_metadata = build_manifest_metadata(
            batch_size=batch_size,
            sample_count=sample_count,
            infer_time=infer_time,
            predictions_path=predictions_path,
        )
    except Exception as e:
        runtime_metadata = {
            "metadata_error": f"Failed to collect runtime metadata: {e}",
        }

    manifest = {
        "model_slug": model_slug,
        "mode": mode,
        "split": split_name,
        "data_path": data_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scores": score_summary or {},
        **runtime_metadata,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    return str(manifest_path)
