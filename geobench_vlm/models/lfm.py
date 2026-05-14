from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from geobench_vlm.utils import build_mcq_prompt, infer_device, infer_dtype

from .base import VLMModel


class LFM25VL(VLMModel):
    """LFM2.5-VL wrapper"""

    model_id = "LiquidAI/LFM2.5-VL-450M"

    def __init__(
        self,
        mode: str = "single",
        model_path: str | None = None,
        max_new_tokens: int = 128,
    ):
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        model_name_or_path = model_path or self.model_id

        # device
        self.device = infer_device()

        # dtype
        dtype = infer_dtype()

        # model
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=dtype,
            attn_implementation="sdpa",
            trust_remote_code=True,
        ).eval()

        # processor
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            # Recommended by https://huggingface.co/LiquidAI/LFM2.5-VL-450M#%F0%9F%8F%83-inference
            min_image_tokens=64,
            max_image_tokens=256,
            do_image_splitting=True,
            trust_remote_code=True,
        )

    @property
    def model_slug(self) -> str:
        return "lfm2_5_vl_450m"

    def build_messages(self, img_path, prompt):
        if self.mode == "temporal":
            if not isinstance(img_path, list) or len(img_path) < 2:
                raise ValueError(
                    f"Temporal mode requires img_path to be a list of at least 2 images. Got: {img_path}"
                )

            pre_img = Image.open(img_path[0]).convert("RGB")
            post_img = Image.open(img_path[-1]).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "The following images show the condition before and after the event.",
                        },
                        {"type": "text", "text": "This is the 'pre' image:"},
                        {"type": "image", "image": pre_img},
                        {"type": "text", "text": "This is the 'post' image:"},
                        {"type": "image", "image": post_img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            image = Image.open(img_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        return messages

    def generate_response(
        self, img_path, question, question_type, options, cls_description
    ) -> str | None:
        if question_type == "Multiple Choice Questions":
            prompt = build_mcq_prompt(question, options, cls_description)
        else:
            prompt = question

        # build messages
        messages = self.build_messages(img_path, prompt)

        # process inputs
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            tokenize=True,
            return_tensors="pt",
        ).to(self.device)

        # generate response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # Deterministic output for evaluation
            use_cache=True,
        )

        #
        generated_trimmed = [
            output_ids[input_ids.shape[0] :]
            for input_ids, output_ids in zip(
                inputs["input_ids"],
                outputs,
            )
        ]

        # decode
        responses = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return responses[0].strip() if responses else None
