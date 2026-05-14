import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
)

from geobench_vlm.utils import (
    build_mcq_prompt,
    infer_attn_impl,
    infer_device,
    infer_dtype,
)

from .base import VLMModel


class LLaVA(VLMModel):
    """Unified LLaVa wrapper"""

    MODEL_CONFIGS = {
        "llava-1.5-7b": {
            "model_id": "llava-hf/llava-1.5-7b-hf",
            "model_cls": LlavaForConditionalGeneration,
            "dtype": torch.float16,
        },
        "llava-1.6-7b": {
            "model_id": "llava-hf/llava-v1.6-vicuna-7b-hf",
            "model_cls": LlavaNextForConditionalGeneration,
            "dtype": torch.bfloat16,
        },
    }

    def __init__(
        self,
        variant: str = "llava-1.6-7b",
        mode: str = "single",
        model_path: str | None = None,
        max_new_tokens: int = 128,
    ):
        if variant not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported LLaVA variant: {variant}. "
                f"Available: {list(self.MODEL_CONFIGS.keys())}"
            )

        self.variant = variant
        self.mode = mode
        self.max_new_tokens = max_new_tokens

        cfg = self.MODEL_CONFIGS[variant]
        self.model_id = cfg["model_id"]

        # model cls
        model_cls = cfg["model_cls"]

        # device
        self.device = infer_device()

        # dtype
        dtype = infer_dtype(cfg["dtype"])
        attn_impl = infer_attn_impl()

        # processor
        model_name_or_path = model_path or self.model_id
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )

        # model
        self.model = model_cls.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_impl,
        ).eval()

    @property
    def model_slug(self) -> str:
        return self.variant.replace("-", "_")

    def build_messages(self, img_path, prompt):
        if self.mode == "temporal":
            raise NotImplementedError(
                "Temporal mode is not implemented for LLaVA models."
            )

        else:
            # LLaVA expects a single image
            image = Image.open(img_path).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                }
            ]

        return image, messages

    def generate_response(
        self,
        img_path,
        question,
        question_type,
        options,
        cls_description,
    ) -> str | None:
        # build prompt
        if question_type == "Multiple Choice Questions":
            prompt = build_mcq_prompt(question, options, cls_description)
        else:
            prompt = question

        # build messages
        image, messages = self.build_messages(
            img_path,
            prompt,
        )

        # chat template
        formatted = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # processor inputs
        inputs = self.processor(
            images=image,
            text=formatted,
            return_tensors="pt",
        ).to(self.device, dtype=self.model.dtype)

        # generate
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,  # Use cache for faster generation
        )

        # remove prompt tokens
        generated_trimmed = output[
            :,
            inputs["input_ids"].shape[1] :,
        ]

        # decode
        response = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response.strip() if response else None
