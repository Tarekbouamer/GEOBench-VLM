from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

pth_m = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'Out_weights', 'LFM2.5-VL-450M')

model = AutoModelForImageTextToText.from_pretrained(
    pth_m, device_map="cuda", torch_dtype=torch.bfloat16).eval()


class MultimodalDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_url = self.dataframe.iloc[idx]['Image']
        question = self.dataframe.iloc[idx]['Tr_Question']
        answer = self.dataframe.iloc[idx]['Tr_Answer']
        question_type = self.dataframe.iloc[idx]['Question_Type']
        question_number = self.dataframe.iloc[idx]['Question_Number']
        ground_truth_option = self.dataframe.iloc[idx]['Ground_Truth_Option']
        options_list = self.dataframe.iloc[idx]['Options_List']
        task = self.dataframe.iloc[idx]['Task']
        question_id = self.dataframe.iloc[idx]['Question_ID']
        cls_description = self.dataframe.iloc[idx]['Cls_Description']
        options = self.dataframe.iloc[idx]['Options']

        return {
            'image_path': image_url,
            'question': question,
            'answer': answer,
            'question_type': question_type,
            'question_number': question_number,
            'ground_truth_option': ground_truth_option,
            'options_list': options_list,
            'task': task,
            'question_id': question_id,
            'cls_description': cls_description,
            'options': options
        }


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def collate_fn(batch):
    return {
        'images': [item['image_path'] for item in batch],
        'questions': [item['question'] for item in batch],
        'answers': [item['answer'] for item in batch],
        'question_type': [item['question_type'] for item in batch],
        'question_number': [item['question_number'] for item in batch],
        'ground_truth_option': [item['ground_truth_option'] for item in batch],
        'options_list': [item['options_list'] for item in batch],
        'task': [item['task'] for item in batch],
        'question_id': [item['question_id'] for item in batch],
        'cls_description': [item['cls_description'] for item in batch],
        'options': [item['options'] for item in batch],
    }


def evaluate(model, dataloader, processor, device):
    model.eval()
    results = []

    with torch.no_grad():
        results_dict = defaultdict(lambda: {
            "predicted_answers": [],
            "ground_truth": None,
            "questions": [],
            "name_images": [],
            "ground_truth_option": None,
            "options_list": None,
            "task": None,
            "question_id": None,
            "cls_description": None,
            "options": None
        })

        for batch in dataloader:
            for img_path, question, answer, question_type, question_number, ground_truth_option, options_list, task, question_id, cls_description, options in zip(
                batch['images'], batch['questions'], batch['answers'], batch['question_type'],
                batch['question_number'], batch['ground_truth_option'], batch['options_list'],
                batch['task'], batch['question_id'], batch['cls_description'], batch['options']
            ):
                try:
                    if question_type == "Multiple Choice Questions":
                        choices = "Options: " + options
                        prompt = (
                            f"For the given the Multiple Choice Question Answer below, analyze the question "
                            f"and answer strictly from one of the options below. Strictly answer the choice only. "
                            f"No additional text. Provide only the letter (A., B., C., D. or E.) corresponding to "
                            f"the correct answer for the multiple-choice question given. {cls_description}\n{question}\n{choices}"
                        )
                    else:
                        prompt = question

                    image = Image.open(img_path).convert("RGB")

                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]

                    print(img_path)

                    inputs = processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        tokenize=True,
                    ).to(device)

                    outputs = model.generate(**inputs, max_new_tokens=128)
                    input_len = inputs["input_ids"].shape[1]
                    predicted_answer = processor.batch_decode(
                        outputs[:, input_len:], skip_special_tokens=True
                    )

                    key = question_number
                    results_dict[key]["predicted_answers"].extend(
                        predicted_answer)
                    results_dict[key]["questions"].append(question)
                    results_dict[key]["ground_truth"] = answer
                    results_dict[key]["name_images"].append(img_path)
                    results_dict[key]["ground_truth_option"] = ground_truth_option
                    results_dict[key]["options_list"] = options_list
                    results_dict[key]["task"] = task
                    results_dict[key]["question_id"] = question_id
                    results_dict[key]["cls_description"] = cls_description
                    results_dict[key]["options"] = options

                except Exception as e:
                    print(f"Error in prediction: {e}")
                    print(f"Question: {question}")

        results.extend(results_dict.values())

    return results


def evaluate_folder(folder_path, max_samples=None):
    qa_file_path = None
    potential_path = os.path.join(folder_path, "Single", "qa.json")
    if os.path.exists(potential_path):
        qa_file_path = potential_path

    if qa_file_path is None:
        print(f"No matching qa file found in {folder_path}. Skipping.")
        return

    with open(qa_file_path, 'r') as file:
        data = json.load(file)
    if max_samples:
        data = data[:max_samples]

    mainp = folder_path
    data_rows = []

    for i, question in enumerate(data):
        image_p = os.path.join(mainp, question.get("image_path", ""))
        ground_truth = question.get("ground_truth", "")
        ground_truth_option = question.get("ground_truth_option", "")
        options_list = question.get("options_list", [])
        task = question.get("task", "")
        question_id = question.get("question_id", "")
        cls_description = question.get("cls_description", "")
        options_str = question.get("options", "")

        for prompt in question.get("prompts", []):
            data_rows.append({
                "Question_Number": i,
                "Category": task,
                "Image": image_p,
                "Question_Type": "Multiple Choice Questions",
                "Tr_Question": prompt,
                "Tr_Answer": ground_truth,
                "Ground_Truth_Option": ground_truth_option,
                "Options_List": options_list,
                "Options": options_str,
                "Task": task,
                "Question_ID": question_id,
                "Cls_Description": cls_description
            })

    df = pd.DataFrame(data_rows)
    dataset = MultimodalDataset(df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32,
                            shuffle=False, collate_fn=collate_fn)

    processor = AutoProcessor.from_pretrained(pth_m)
    scores = evaluate(model, dataloader, processor, device)

    result_folder = os.path.join(folder_path, "Results-lfm25-vl")
    os.makedirs(result_folder, exist_ok=True)

    result_file = os.path.join(
        result_folder, f"evaluation_results_{os.path.basename(folder_path)}.json")
    result_filet = os.path.join(
        result_folder, f"evaluation_results_{os.path.basename(folder_path)}.txt")

    try:
        with open(result_file, "w") as f:
            json.dump(scores, f, indent=4, default=str)
    except Exception as e:
        print(f"Error in saving results: {e}")
        with open(result_filet, "w") as f:
            f.write(str(scores))

    print(f"Results saved successfully for folder {folder_path}.")


def main(base_folder_path, max_samples=None):
    folder_path = base_folder_path
    print(folder_path)
    if os.path.isdir(folder_path):
        evaluate_folder(folder_path, max_samples)
    else:
        print(f"{folder_path} is not a directory. Skipping.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/datasets/GEOBench-VLM")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of questions (for smoke testing)")
    args = parser.parse_args()
    main(args.data_path, args.max_samples)
