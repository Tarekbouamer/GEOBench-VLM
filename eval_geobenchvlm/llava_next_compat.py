import importlib
import os
import sys
import types


def get_llava_next_dir(explicit_path=None):
    if explicit_path:
        return explicit_path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "LLaVA-NeXT")


def _clear_llava_modules():
    for module_name in list(sys.modules):
        if module_name == "llava" or module_name.startswith("llava."):
            sys.modules.pop(module_name, None)


def _bootstrap_llava_namespace(llava_next_dir):
    package_dir = os.path.join(llava_next_dir, "llava")
    llava_module = types.ModuleType("llava")
    llava_module.__file__ = os.path.join(package_dir, "__init__.py")
    llava_module.__path__ = [package_dir]
    llava_module.__package__ = "llava"
    sys.modules["llava"] = llava_module


def ensure_llava_next_importable(llava_next_dir=None):
    llava_next_dir = get_llava_next_dir(llava_next_dir)
    if not os.path.isdir(llava_next_dir):
        raise FileNotFoundError(
            f"LLaVA-NeXT repository not found at {llava_next_dir}"
        )

    if llava_next_dir not in sys.path:
        sys.path.insert(0, llava_next_dir)
    importlib.invalidate_caches()

    try:
        importlib.import_module("llava.model.builder")
        return llava_next_dir
    except Exception as first_error:
        _clear_llava_modules()
        _bootstrap_llava_namespace(llava_next_dir)
        importlib.invalidate_caches()
        try:
            importlib.import_module("llava.model.builder")
            return llava_next_dir
        except Exception as second_error:
            raise RuntimeError(
                "Unable to import LLaVA-NeXT inference modules. "
                f"Standard import failed with: {first_error}. "
                f"Namespace bootstrap failed with: {second_error}."
            ) from second_error


def load_llava_runtime(llava_next_dir=None):
    llava_next_dir = ensure_llava_next_importable(llava_next_dir)

    builder = importlib.import_module("llava.model.builder")
    mm_utils = importlib.import_module("llava.mm_utils")
    constants = importlib.import_module("llava.constants")
    conversation = importlib.import_module("llava.conversation")

    return {
        "llava_next_dir": llava_next_dir,
        "load_pretrained_model": builder.load_pretrained_model,
        "process_images": mm_utils.process_images,
        "tokenizer_image_token": mm_utils.tokenizer_image_token,
        "IMAGE_TOKEN_INDEX": constants.IMAGE_TOKEN_INDEX,
        "DEFAULT_IMAGE_TOKEN": constants.DEFAULT_IMAGE_TOKEN,
        "conv_templates": conversation.conv_templates,
    }
