# GEOBench-VLM Evaluation Guide

## Setup

Follow the [Installation](README.md#-installation) section in the README to set up the environment with `uv`.

## Model Weights

Download the pretrained model weights and place them under `Out_weights/`:

| Model | HuggingFace Link |
|---|---|
| LLaVA 1.5 (Vicuna-7B) | [llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) |
| LLaVA 1.6 (Vicuna-7B) | [llava-v1.6-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf) |
| InternVL2 (8B) | [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B) |
| Qwen2-VL (7B Instruct) | [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) |
| LLaVA-OneVision | [llava-onevision-qwen2-7b-si](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-si) |

```
Out_weights/
├── llava-1.5-7b-hf/
├── llava-v1.6-vicuna-7b-hf/
├── InternVL2-8B/
├── Qwen2-VL-7B-Instruct/
└── llava-onevision-qwen2-7b-si/
```

---

## Single-Image Evaluation

```bash
uv run geobench-single --model <model-key> --data <path/to/split> --results <output/dir>
```

| Model | Key |
|---|---|
| LLaVA 1.5 | `llava1pt5` |
| LLaVA 1.6 | `llava1pt6` |
| LLaVA-OneVision | `llavaone1` |
| Qwen2-VL | `qwen` |
| InternVL2 | `internvl` |
| LFM-2.5 | `lfm` |

**Example:**

```bash
uv run geobench-single --model qwen --data data/GEOBench-VLM --results results/
```

**Or use the provided scripts** (set `DATA_PATH` to your dataset location):

```bash
DATA_PATH=/path/to/GEOBench-VLM bash scripts/eval_qwen.sh
DATA_PATH=/path/to/GEOBench-VLM bash scripts/eval_lfm.sh
DATA_PATH=/path/to/GEOBench-VLM bash scripts/eval_llavaone1.sh
```

**Optional flags:**

| Flag | Description |
|---|---|
| `--score` / `-s` | Compute and print accuracy after inference |
| `--max-samples N` | Limit to N questions (smoke test) |
| `--batch-size N` | DataLoader batch size (default: 32) |

---

## Temporal Evaluation

```bash
uv run geobench-temporal --model <model-key> --data <path/to/split> --results <output/dir>
```

| Model | Key |
|---|---|
| Qwen2-VL | `qwen` |
| LLaVA-OneVision | `llavaone1` |

**Example:**

```bash
uv run geobench-temporal --model qwen --data data/GEOBench-VLM --results results/
```

---

## Scoring

Scoring is no longer exposed as a CLI command in this repository.
