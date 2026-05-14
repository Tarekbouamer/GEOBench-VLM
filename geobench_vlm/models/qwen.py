try:
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None
    AutoProcessor = None


try:
    from qwen_vl_utils import process_vision_info as qwen_process_vision_info
except ImportError:

    def qwen_process_vision_info(messages):
        return None, None


from geobench_vlm.utils import (
    build_mcq_prompt,
    infer_attn_impl,
    infer_device,
    infer_dtype,
)

from .base import VLMModel


class Qwen2VL(VLMModel):
    """Qwen2-VL wrapper"""

    model_id = "Qwen/Qwen2-VL-7B-Instruct"

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

        # attn
        attn_impl = infer_attn_impl()
        # model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        ).eval()

        # processor
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

    @property
    def model_slug(self) -> str:
        return self.model_id.split("/")[-1].lower().replace("-", "_")

    def build_messages(self, img_path, prompt):
        """Build messages for Qwen2VL"""

        if self.mode == "temporal":
            if not isinstance(img_path, list) or len(img_path) < 2:
                raise ValueError(
                    f"Temporal mode requires img_path to be a list of 2 images. Got: {img_path}"
                )

            pre_img = img_path[0]
            post_img = img_path[-1]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "The following images show the condition before and after the event.",
                        },
                        {"type": "text", "text": "This is the 'pre' image:"},
                        {"type": "image", "image": f"file://{pre_img}"},
                        {"type": "text", "text": "This is the 'post' image:"},
                        {"type": "image", "image": f"file://{post_img}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{img_path}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        return messages

    def generate_response(
        self, img_path, question, question_type, options, cls_description
    ) -> str | None:
        # Build prompt
        if question_type == "Multiple Choice Questions":
            prompt = build_mcq_prompt(question, options, cls_description)
        else:
            prompt = question

        # build messages
        messages = self.build_messages(img_path, prompt)

        # process inputs format
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Vision processing
        # This ensures the image is compatible with the ViT encoder’s patch processing logic.
        image_inputs, video_inputs = qwen_process_vision_info(messages)

        # process inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # generate response
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # as temperature =0.0 deterministic decoding
            use_cache=True,
        )

        # remove prompt tokens
        generated_trimmed = [
            output_ids[input_ids.shape[0] :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode the thing
        responses = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return responses[0].strip() if responses else None
