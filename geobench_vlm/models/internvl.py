from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

from geobench_vlm.utils import (
    build_mcq_prompt,
    infer_attn_impl,
    infer_device,
    infer_dtype,
)

from .base import VLMModel

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int = 448):
    return transforms.Compose(
        [
            transforms.Resize(
                (input_size, input_size),
                interpolation=Image.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def _load_image(
    image_path: str,
    image_size: int = 448,
):
    image = Image.open(image_path).convert("RGB")
    transform = _build_transform(image_size)
    return transform(image).unsqueeze(0)


class InternVL2(VLMModel):
    """InternVL2 Wrapper"""

    model_id = "OpenGVLab/InternVL2-8B"

    def __init__(
        self,
        mode: str = "single",
        model_path: str | None = None,
        max_new_tokens: int = 128,
        image_size: int = 448,
    ):
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.image_size = image_size

        # device
        self.device = infer_device()

        # dtype
        dtype = infer_dtype()

        # flash attention support
        use_flash_attn = infer_attn_impl() == "flash_attention_2"

        # model
        model_name_or_path = model_path or self.model_id

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )

    @property
    def model_slug(self) -> str:
        return "internvl2_8b"

    def build_messages(self, img_path, prompt):
        if self.mode == "temporal":
            if not isinstance(img_path, list) or len(img_path) < 2:
                raise ValueError(
                    "Temporal mode requires img_path to be a list of at least 2 images."
                )
            raise NotImplementedError(
                "Temporal mode is not yet implemented for InternVL2. "
                "InternVL2 requires a dedicated multi-image preprocessing "
                "pipeline for temporal reasoning."
            )
        else:
            pixel_values = _load_image(
                img_path,
                self.image_size,
            ).to(dtype=self.model.dtype, device=self.device)

            message = prompt

        return pixel_values, message

    def generate_response(
        self,
        img_path,
        question,
        question_type,
        options,
        cls_description,
    ) -> str | None:
        # Build prompt
        if question_type == "Multiple Choice Questions":
            prompt = build_mcq_prompt(
                question,
                options,
                cls_description,
            )
        else:
            prompt = question

        # build message
        pixel_values, message = self.build_messages(img_path, prompt)

        # generate response
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            message,
            {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
            },
        )

        # cleanup
        if isinstance(response, str):
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()

            response = response.strip()

        return response if response else None
