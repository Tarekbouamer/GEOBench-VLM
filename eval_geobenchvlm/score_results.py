import argparse
import collections
import json
import os

MODEL_RESULT_FOLDERS = {
    "Results-qwen2-countcls":         "Qwen2-VL-7B",
    "Results-lfm25-vl":               "LFM2.5-VL-450M",
    "Results-llava1pt5-countcls":     "LLaVA-1.5-7B",
    "Results-llava1pt6-countcls":     "LLaVA-1.6-7B",
    "Results-llavaOne-countcls":      "LLaVA-OneVision-7B",
    "Results-llavaInternVL-countcls": "InternVL2-8B",
}


def extract_letter(pred: str) -> str:
    """Return the first A-E letter found in the prediction string."""
    pred = pred.strip().upper()
    for ch in pred:
        if ch in "ABCDE":
            return ch
    return ""


def score_results(results: list) -> tuple[dict, dict]:
    """Return (task_correct, task_total) dicts from a results list."""
    task_correct: dict[str, int] = collections.defaultdict(int)
    task_total:   dict[str, int] = collections.defaultdict(int)

    for entry in results:
        gt = str(entry.get("ground_truth_option", "")).strip().upper()
        if not gt:
            continue
        task = str(entry.get("task", "Unknown"))

        predicted_answers = entry.get("predicted_answers", [])
        if not predicted_answers:
            task_total[task] += 1
            continue

        votes = [extract_letter(p) for p in predicted_answers]
        votes = [v for v in votes if v]
        majority = collections.Counter(votes).most_common(1)[
            0][0] if votes else ""

        task_total[task] += 1
        if majority == gt:
            task_correct[task] += 1

    return dict(task_correct), dict(task_total)


def find_result_files(data_path: str) -> dict[str, str]:
    """Return {model_name: json_file_path} for every results folder found."""
    found = {}
    for folder_name, model_name in MODEL_RESULT_FOLDERS.items():
        folder_path = os.path.join(data_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if fname.endswith(".json"):
                found[model_name] = os.path.join(folder_path, fname)
                break   
    return found


def build_table(all_scores: dict) -> str:
    """
    all_scores: {model_name: {"per_task": {task: acc}, "overall": acc}}
    Returns a nicely formatted ASCII table.
    """
    all_tasks = sorted({
        task
        for scores in all_scores.values()
        for task in scores["per_task"]
    })
    model_names = sorted(all_scores.keys())

    task_col_w = max(len(t) for t in all_tasks + ["Task"]) + 2
    model_col_w = max(max(len(m) for m in model_names), 8) + 2

    header = f"{'Task':<{task_col_w}}" + \
        "".join(f"{m:>{model_col_w}}" for m in model_names)
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
    lines.append(overall_row)
    lines.append(sep)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                        help="Root dataset folder (contains Results-* subfolders)")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save scores.json and scores.csv (default: data_path)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_path
    os.makedirs(output_dir, exist_ok=True)

    result_files = find_result_files(args.data_path)
    if not result_files:
        print("No result JSON files found. Run eval scripts first.")
        return

    all_scores = {}

    for model_name, json_path in sorted(result_files.items()):
        print(f"Scoring {model_name} from {json_path} ...")
        with open(json_path) as f:
            results = json.load(f)

        task_correct, task_total = score_results(results)

        per_task = {}
        for task in sorted(task_total):
            correct = task_correct.get(task, 0)
            total = task_total[task]
            per_task[task] = round(100 * correct / total, 2) if total else 0.0

        total_correct = sum(task_correct.values())
        total_all = sum(task_total.values())
        overall = round(100 * total_correct / total_all,
                        2) if total_all else 0.0

        all_scores[model_name] = {"per_task": per_task, "overall": overall}

    # --- Print table ---
    table = build_table(all_scores)
    print("\n" + table)

    # --- Save JSON ---
    json_out = os.path.join(output_dir, "scores.json")
    with open(json_out, "w") as f:
        json.dump(all_scores, f, indent=4)
    print(f"\nScores saved to {json_out}")

    # --- Save CSV ---
    all_tasks = sorted({
        task
        for scores in all_scores.values()
        for task in scores["per_task"]
    })
    model_names = sorted(all_scores.keys())

    csv_out = os.path.join(output_dir, "scores.csv")
    with open(csv_out, "w") as f:
        f.write("Task," + ",".join(model_names) + "\n")
        for task in all_tasks:
            row = [task]
            for m in model_names:
                acc = all_scores[m]["per_task"].get(task, "")
                row.append(str(acc))
            f.write(",".join(row) + "\n")
        overall_row = ["OVERALL"]
        for m in model_names:
            overall_row.append(str(all_scores[m].get("overall", "")))
        f.write(",".join(overall_row) + "\n")
    print(f"Scores saved to {csv_out}")


if __name__ == "__main__":
    main()
