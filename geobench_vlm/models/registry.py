import importlib


def _make_lazy(module: str, cls: str, **lazy_kwargs):
    """Return a proxy class that lazy-imports *cls* from the given sub-module."""

    class _Lazy:
        def __new__(cls_, *args, **kwargs):
            merged_kwargs = {**lazy_kwargs, **kwargs}
            mod = importlib.import_module(f"geobench_vlm.models.{module}")
            return getattr(mod, cls)(*args, **merged_kwargs)

    _Lazy.__name__ = cls
    _Lazy.__qualname__ = cls
    return _Lazy


_LazyQwen2VL = _make_lazy("qwen", "Qwen2VL")
_LazyLFM25VL = _make_lazy("lfm", "LFM25VL")
_LazyLLaVA15 = _make_lazy("llava", "LLaVA", variant="llava-1.5-7b")
_LazyLLaVA16 = _make_lazy("llava", "LLaVA", variant="llava-1.6-7b")
_LazyInternVL2 = _make_lazy("internvl", "InternVL2")
_LazyLLaVAOneVision = _make_lazy("llavaone1", "LLaVAOneVision")


SINGLE_MODELS: dict[str, type] = {
    "qwen": _LazyQwen2VL,
    "lfm": _LazyLFM25VL,
    "llavaone1": _LazyLLaVAOneVision,
    "llava1pt5": _LazyLLaVA15,
    "llava1pt6": _LazyLLaVA16,
    "internvl": _LazyInternVL2,
}

TEMPORAL_MODELS: dict[str, type] = {
    "qwen": _LazyQwen2VL,
    "llavaone1": _LazyLLaVAOneVision,
}
