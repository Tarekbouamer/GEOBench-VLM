import copy

import torch

# Shim: apply_chunking_to_forward was removed in transformers 4.45 but LLaVA-NeXT still uses it.
import transformers.modeling_utils as _transformers_mu
from PIL import Image

if not hasattr(_transformers_mu, "apply_chunking_to_forward"):
    from geobench_vlm.utils.transformers_compat import apply_chunking_to_forward

    _transformers_mu.apply_chunking_to_forward = apply_chunking_to_forward

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

from geobench_vlm.utils import build_mcq_prompt, infer_device

from .base import VLMModel

_CONV_TEMPLATE = "qwen_1_5"


class LLaVAOneVision(VLMModel):
    """LLaVA-OneVision-7B model for single and temporal evaluation."""

    model_id = "lmms-lab/llava-onevision-qwen2-7b-si"

    def __init__(self, mode: str = "single", model_path: str | None = None):
        self.mode = mode
        pth = model_path or self.model_id
        self.device = infer_device()
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            pth,
            None,
            "llava_qwen",
            device_map=self.device,
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.model.to(self.device)

    @property
    def model_slug(self) -> str:
        return "llava-onevision-7b"

    def _build_inputs(self, pil_images: list, prompt: str):
        image_tensors = process_images(
            pil_images, self.image_processor, self.model.config
        )
        image_tensors = [
            img.to(dtype=torch.float16, device=self.device) for img in image_tensors
        ]

        if self.mode == "temporal":
            token_prompt = (
                f"{DEFAULT_IMAGE_TOKEN} This is the 'pre' image.\n\n"
                f"Now, let's look at this image. This is the 'post' image. {DEFAULT_IMAGE_TOKEN}\n\n{prompt}"
            )
        else:
            token_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"

        conv = copy.deepcopy(conv_templates[_CONV_TEMPLATE])
        conv.append_message(conv.roles[0], token_prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.device)
        )

        image_sizes = [img.size for img in pil_images]
        return input_ids, image_tensors, image_sizes

    def generate_response(
        self, img_path, question, question_type, options, cls_description
    ) -> str | None:
        if question_type == "Multiple Choice Questions":
            prompt = build_mcq_prompt(question, options, cls_description)
        else:
            prompt = question

        if self.mode == "temporal":
            paths = img_path if isinstance(img_path, list) else [img_path, img_path]
            pil_images = [Image.open(p).convert("RGB") for p in paths]
        else:
            pil_images = [Image.open(img_path).convert("RGB")]

        input_ids, image_tensors, image_sizes = self._build_inputs(pil_images, prompt)

        cont = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
        )
        responses = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        response = responses[0] if responses else ""

        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        return response if response else None
