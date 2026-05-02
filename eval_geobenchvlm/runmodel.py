#!/usr/bin/env python3
"""runmodel.py — dispatch eval to a model module and call main() directly.

Usage
-----
python runmodel.py <model-key> --data_path PATH --results_dir PATH [--max_samples N]

Model keys
----------
llava1pt5 | llava1pt6 | llavaone1 | qwen | internvl | lfm
"""

import argparse
import importlib
import os
import sys

MODEL2MODULE = {
    "llava1pt5": "llava1pt5_cls_single",
    "llava1pt6": "llava1pt6_cls_single",
    "llavaone1": "llavaone1_cls_single",
    "qwen":      "qwen_cls_single",
    "internvl":  "internvl_cls_single",
    "lfm":       "lfm_cls_single",
}

# Ensure the directory containing this file is on sys.path so imports work
# whether this script is run directly or imported from a notebook.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def run(model_key: str, data_path: str, results_dir: str, max_samples=None):
    module_name = MODEL2MODULE.get(model_key)
    if module_name is None:
        raise ValueError(
            f"Unknown model key '{model_key}'. Choose from: {', '.join(MODEL2MODULE)}")
    mod = importlib.import_module(module_name)
    mod.main(data_path, results_dir, max_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key", choices=MODEL2MODULE)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    run(args.model_key, args.data_path, args.results_dir, args.max_samples)
