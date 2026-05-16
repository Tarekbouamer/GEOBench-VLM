from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def run_eval(model, dataloader: DataLoader, device: torch.device) -> list[dict]:
    """Run inference over a DataLoader and aggregate predictions by question number."""
    results_dict: dict = defaultdict(
        lambda: {
            "predicted_answers": [],
            "ground_truth": None,
            "questions": [],
            "name_images": [],
            "ground_truth_option": None,
            "options_list": None,
            "task": None,
            "question_id": None,
            "cls_description": None,
            "options": None,
        }
    )

    slug = getattr(model, "model_slug", "model")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=slug, unit="batch", dynamic_ncols=True):
            items = zip(
                batch["image_path"],
                batch["question"],
                batch["answer"],
                batch["question_type"],
                batch["question_number"],
                batch["ground_truth_option"],
                batch["options_list"],
                batch["task"],
                batch["question_id"],
                batch["cls_description"],
                batch["options"],
            )
            for (
                img_path,
                question,
                answer,
                question_type,
                question_number,
                ground_truth_option,
                options_list,
                task,
                question_id,
                cls_description,
                options,
            ) in items:
                try:
                    predicted_answer = model.generate_response(
                        img_path=img_path,
                        question=question,
                        question_type=question_type,
                        options=options,
                        cls_description=cls_description,
                    )
                except Exception as e:
                    raise RuntimeError(
                        "Inference failed "
                        f"image={img_path} question_id={question_id} task={task}"
                    ) from e

                key = question_number
                if predicted_answer is not None:
                    if isinstance(predicted_answer, list):
                        results_dict[key]["predicted_answers"].extend(predicted_answer)
                    else:
                        results_dict[key]["predicted_answers"].append(predicted_answer)
                results_dict[key]["questions"].append(question)
                results_dict[key]["ground_truth"] = answer
                results_dict[key]["name_images"].append(img_path)
                results_dict[key]["ground_truth_option"] = ground_truth_option
                results_dict[key]["options_list"] = options_list
                results_dict[key]["task"] = task
                results_dict[key]["question_id"] = question_id
                results_dict[key]["cls_description"] = cls_description
                results_dict[key]["options"] = options

    return list(results_dict.values())
