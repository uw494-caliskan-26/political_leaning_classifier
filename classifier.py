import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoTokenizer, BitsAndBytesConfig

if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

load_dotenv()

MODEL_ID = "zhezhou1106/political-leaning-classifier-v2"
LABEL_MAP = {
    0: "Far Left",
    1: "Lean Left",
    2: "Center",
    3: "Lean Right",
    4: "Far Right",
}

DEFAULT_INFERENCE_BATCH_SIZE = 8

PROMPT_TEMPLATE = """You are an expert media bias analyst. Read the following news article and classify its political leaning on a 0-4 scale.

Article:
{article}

Scale:
class 0: extremely left (far left)
class 1: lean left
class 2: center / neutral
class 3: lean right
class 4: extremely right (far right)

SOLUTION
The correct answer is: class """


def _remap_key(key: str) -> str:
    """Remap state dict keys from Unsloth's nested structure to standard transformers."""
    if key.startswith("model.language_model.language_model.language_model."):
        return (
            "model.language_model."
            + key[len("model.language_model.language_model.language_model.") :]
        )
    if key.startswith("model.language_model.visual."):
        return "model.visual." + key[len("model.language_model.visual.") :]
    return key


def _needs_key_remap(model_path: str) -> bool:
    """Check if the local model's safetensors have Unsloth's nested key format."""
    safetensors_path = Path(model_path) / "model.safetensors"
    if not safetensors_path.exists():
        return False
    try:
        from safetensors import safe_open

        with safe_open(str(safetensors_path), framework="pt") as f:
            keys = f.keys()
            return any(
                k.startswith("model.language_model.language_model.language_model.")
                for k in keys
            )
    except Exception:
        return False


def _get_remapped_model_path(model_path: str) -> str:
    """Create a cached copy of the model with remapped safetensors keys."""
    cache_dir = Path(model_path) / ".remapped_cache"
    remapped_safetensors = cache_dir / "model.safetensors"

    if remapped_safetensors.exists():
        return str(cache_dir)

    print("Detected Unsloth key format. Remapping state dict keys (one-time)...")
    from safetensors.torch import load_file, save_file

    state_dict = load_file(str(Path(model_path) / "model.safetensors"))
    new_state_dict = {_remap_key(k): v for k, v in state_dict.items()}

    cache_dir.mkdir(parents=True, exist_ok=True)
    save_file(new_state_dict, str(remapped_safetensors))

    for f in Path(model_path).iterdir():
        if f.name in ("model.safetensors", ".remapped_cache", ".cache"):
            continue
        dest = cache_dir / f.name
        if not dest.exists():
            shutil.copy2(str(f), str(dest))

    print("Key remapping complete.")
    return str(cache_dir)


def load_model(quantize_4bit: bool = True):
    model_path = os.getenv("MODEL_PATH", MODEL_ID)

    is_local = Path(model_path).is_dir()
    if is_local and _needs_key_remap(model_path):
        model_path = _get_remapped_model_path(model_path)

    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except ImportError:
        from transformers import AutoModelForCausalLM as AutoModelCls

    _BNB_SKIP_MODULES = ["model.visual", "lm_head"]

    kwargs = {
        "trust_remote_code": True,
    }

    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = None

    if quantize_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=list(_BNB_SKIP_MODULES),
        )
    else:
        kwargs["dtype"] = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelCls.from_pretrained(model_path, **kwargs)
    model.eval()

    number_token_ids = []
    for digit in range(5):
        ids = tokenizer.encode(str(digit), add_special_tokens=False)
        assert len(ids) == 1, (
            f"Tokenizer must encode digit '{digit}' as a single token."
        )
        number_token_ids.append(ids[0])

    return model, tokenizer, number_token_ids


def classify(text: str, model, tokenizer, number_token_ids) -> dict:
    prompt = PROMPT_TEMPLATE.format(article=text.strip())
    enc = tokenizer([prompt], return_tensors="pt", padding=True).to(model.device)

    with torch.inference_mode():
        out = model(**enc, logits_to_keep=1)
        last_logits = out.logits[:, -1, :]
        digit_logits = last_logits[:, number_token_ids]
        probs = F.softmax(digit_logits, dim=-1)

    score = int(probs.argmax(dim=-1).item())
    prob_list = probs[0].tolist()
    weighted_score = sum(i * p for i, p in enumerate(prob_list))

    return {
        "score": score,
        "weighted_score": round(weighted_score, 4),
        "label": LABEL_MAP[score],
        "probabilities": {LABEL_MAP[i]: round(p, 4) for i, p in enumerate(prob_list)},
    }


def classify_batch(
    texts: list[str],
    model,
    tokenizer,
    number_token_ids,
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
) -> list[dict]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    results: list[dict] = []
    device = model.device

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        prompts = [PROMPT_TEMPLATE.format(article=t.strip()) for t in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        with torch.inference_mode():
            out = model(**enc, logits_to_keep=1)
            last_logits = out.logits[:, -1, :]
            digit_logits = last_logits[:, number_token_ids]
            probs = F.softmax(digit_logits, dim=-1)

        for i in range(probs.shape[0]):
            prob_list = probs[i].tolist()
            score = int(probs[i].argmax().item())
            weighted_score = sum(j * p for j, p in enumerate(prob_list))
            results.append(
                {
                    "score": score,
                    "weighted_score": round(weighted_score, 4),
                    "label": LABEL_MAP[score],
                    "probabilities": {
                        LABEL_MAP[j]: round(prob_list[j], 4) for j in range(5)
                    },
                }
            )

    return results
