import importlib.util

import torch


def support_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def support_flash_attn() -> bool:
    if not torch.cuda.is_available():
        return False

    # at least 8
    major, _ = torch.cuda.get_device_capability()

    # check install
    flash_attn_installed = importlib.util.find_spec("flash_attn") is not None

    return major >= 8 and flash_attn_installed


def infer_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_dtype(torch_dtype: torch.dtype = torch.bfloat16) -> torch.dtype:
    if torch_dtype == torch.bfloat16 and not support_bf16():
        return torch.float16

    return torch_dtype


def infer_attn_impl() -> str:
    return "flash_attention_2" if support_flash_attn() else "sdpa"
