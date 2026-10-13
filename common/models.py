"""Loading a model variant. Every script goes through load_variant so that
perplexity, footprint and lm-eval all see identically loaded models."""
from common.config import model_dir

QUANT_METHODS = {"gptq", "bnb_nf4"}


def variant_path(cfg, variant):
    """All variants, including the fp16 baseline, live in models/<slug>/<variant>."""
    return str(model_dir(cfg, variant))


def load_variant(cfg, variant, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = variant_path(cfg, variant)
    # fp16 everywhere: the T4 has no bf16 support.
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map=device
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok


def lm_eval_model_args(cfg, variant):
    """model_args string for lm_eval's hf backend. Quantized checkpoints carry their
    quantization_config in config.json, so loading by path is enough for all variants."""
    return f"pretrained={variant_path(cfg, variant)},dtype=float16"
