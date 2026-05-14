import json
import os
from pathlib import Path

from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    """Loads qa.json and builds one row per prompt for single or temporal evaluation."""

    def __init__(
        self, data_path: str, mode: str = "single", max_samples: int | None = None
    ):
        subdir = "Single" if mode == "single" else "Temporal"
        qa_path = Path(data_path) / subdir / "qa.json"
        if not qa_path.exists():
            raise FileNotFoundError(
                f"No qa.json found at {qa_path}. "
                f"Expected layout: {data_path}/{subdir}/qa.json"
            )

        with open(qa_path) as f:
            data = json.load(f)

        if max_samples:
            data = data[:max_samples]

        self.rows: list[dict] = []
        for i, q in enumerate(data):
            raw_path = q.get("image_path", "")

            if mode == "temporal":
                if isinstance(raw_path, list):
                    image_p = [
                        os.path.join(data_path, raw_path[0]),
                        os.path.join(data_path, raw_path[-1]),
                    ]
                else:
                    image_p = os.path.join(data_path, raw_path)
            else:
                image_p = os.path.join(data_path, raw_path)

            base = {
                "image_path": image_p,
                "question_number": i,
                "question_type": "Multiple Choice Questions",
                "answer": q.get("ground_truth", ""),
                "ground_truth_option": q.get("ground_truth_option", ""),
                "options_list": q.get("options_list", []),
                "options": q.get("options", ""),
                "task": q.get("task", ""),
                "question_id": q.get("question_id", ""),
                "cls_description": q.get("cls_description", ""),
            }

            for prompt in q.get("prompts", []):
                self.rows.append({**base, "question": prompt})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def collate_fn(batch: list[dict]) -> dict:
    """Default collate: gather each field into a list, no tensor conversion."""
    keys = batch[0].keys()
    return {k: [item[k] for item in batch] for k in keys}
