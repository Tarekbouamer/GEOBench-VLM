from abc import ABC, abstractmethod


class VLMModel(ABC):
    """VLM Base Model"""

    @property
    @abstractmethod
    def model_slug(self) -> str:
        """Short identifier used for result folder naming, e.g. 'qwen2-vl-7b'."""

    @abstractmethod
    def generate_response(
        self,
        img_path,
        question: str,
        question_type: str,
        options: str,
        cls_description: str,
    ) -> str | None:
        """Run one forward pass and return the raw model response string."""
