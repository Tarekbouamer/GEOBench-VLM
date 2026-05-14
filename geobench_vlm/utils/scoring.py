import collections
import os
import re as _re


def extract_letter(pred: str = "") -> str:
    """Return the best A-E answer letter from *pred*, or empty string."""

    # Prefer a standalone token: "A.", "A)", "(A)", "A:", "A " at word boundary
    m = _re.search(r"(?<![A-Z])([A-E])(?=[.):\s]|$)", pred.strip().upper())
    if m:
        return m.group(1)
    for ch in pred.strip().upper():
        if ch in "ABCDE":
            return ch
    return ""


def score_results(results: list[dict]) -> tuple[dict, dict]:
    """
    Compute per-task correct/total counts.

    Returns:
        (task_correct, task_total) — both defaultdict(int).
    """
    task_correct: dict[str, int] = collections.defaultdict(int)
    task_total: dict[str, int] = collections.defaultdict(int)

    for entry in results:
        gt = str(entry.get("ground_truth_option", "")).strip().upper()
        if not gt:
            continue
        task = str(entry.get("task", "Unknown"))

        predicted_answers = entry.get("predicted_answers", [])
        if not predicted_answers:
            task_total[task] += 1
            continue

        votes = [extract_letter(p) for p in predicted_answers if isinstance(p, str)]
        votes = [v for v in votes if v]
        majority = collections.Counter(votes).most_common(1)[0][0] if votes else ""

        task_total[task] += 1
        if majority == gt:
            task_correct[task] += 1

    return dict(task_correct), dict(task_total)


def build_score_summary(results: list[dict]) -> dict:
    """
    Return a summary dict with per_task accuracies and overall accuracy.
    Suitable for embedding in manifest.json.
    """
    task_correct, task_total = score_results(results)

    per_task = {}
    for task in sorted(task_total):
        correct = task_correct.get(task, 0)
        total = task_total[task]
        per_task[task] = round(100 * correct / total, 2) if total else 0.0

    total_correct = sum(task_correct.values())
    total_all = sum(task_total.values())
    overall = round(100 * total_correct / total_all, 2) if total_all else 0.0

    return {"per_task": per_task, "overall": overall}


def find_result_files(results_dir: str, mode: str = "single") -> dict[str, list[str]]:
    """
    Scan results_dir for model-slug subdirs containing a {mode}/ subfolder.

    Returns {model_slug: [json_path, ...]} for every model found.
    """
    found: dict[str, list[str]] = {}
    if not os.path.isdir(results_dir):
        return {}
    for slug in sorted(os.listdir(results_dir)):
        mode_dir = os.path.join(results_dir, slug, mode)
        if not os.path.isdir(mode_dir):
            continue
        json_files = sorted(
            f
            for f in os.listdir(mode_dir)
            if f.endswith(".json") and f != "manifest.json"
        )
        if json_files:
            found[slug] = [os.path.join(mode_dir, f) for f in json_files]
    return found


def build_table(all_scores: dict) -> str:
    """Render an ASCII comparison table from {model: {per_task, overall}}."""
    all_tasks = sorted({t for s in all_scores.values() for t in s["per_task"]})
    model_names = sorted(all_scores.keys())

    task_col_w = max((len(t) for t in all_tasks + ["Task"]), default=4) + 2
    model_col_w = max((len(m) for m in model_names), default=8) + 2

    header = f"{'Task':<{task_col_w}}" + "".join(
        f"{m:>{model_col_w}}" for m in model_names
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for task in all_tasks:
        row = f"{task:<{task_col_w}}"
        for m in model_names:
            acc = all_scores[m]["per_task"].get(task)
            cell = f"{acc:.1f}%" if acc is not None else "  n/a"
            row += f"{cell:>{model_col_w}}"
        lines.append(row)

    lines.append(sep)
    overall_row = f"{'OVERALL':<{task_col_w}}"
    for m in model_names:
        acc = all_scores[m].get("overall")
        cell = f"{acc:.1f}%" if acc is not None else "  n/a"
        overall_row += f"{cell:>{model_col_w}}"
    lines += [overall_row, sep]

    return "\n".join(lines)
