"""GPTQ 4-bit quantization with GPTQModel. Calibration text comes from the wikitext-2
train split (perplexity is measured on the test split)."""
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from common.config import base_parser, load_config, model_dir, results_dir, set_seed
from common.metrics import gpu_info, weights_size_bytes, write_json


def calibration_texts(cfg, tokenizer):
    """num_samples fixed-length chunks (seq_len tokens) from consecutive train text, seeded."""
    c = cfg["calibration"]
    ds = load_dataset(c["dataset"], c["config"], split=c["split"])
    ids = tokenizer("\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt").input_ids[0]
    n, L = c["num_samples"], c["seq_len"]
    g = torch.Generator().manual_seed(cfg["seed"])
    starts = torch.randint(0, len(ids) - L - 1, (n,), generator=g)
    return [tokenizer.decode(ids[s : s + L]) for s in starts.tolist()]


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    v = cfg["variants"]["gptq"]
    src, out = model_dir(cfg, "fp16"), model_dir(cfg, "gptq")

    from gptqmodel import GPTQModel
    try:  # class name differs between gptqmodel releases
        from gptqmodel import GPTQConfig as QCfg
    except ImportError:
        from gptqmodel import QuantizeConfig as QCfg

    tok = AutoTokenizer.from_pretrained(src)
    texts = calibration_texts(cfg, tok)

    qcfg = QCfg(bits=v["bits"], group_size=v["group_size"], desc_act=v["desc_act"], sym=v["sym"])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = GPTQModel.load(str(src), qcfg)
    model.quantize(texts, batch_size=1)
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated()

    model.save(str(out))
    tok.save_pretrained(out)

    rec = {
        "variant": "gptq",
        "config": v,
        "calibration": cfg["calibration"],
        "quantize_seconds": round(elapsed, 1),
        "peak_gpu_bytes_during_quantization": peak,
        "weights_bytes": weights_size_bytes(out),
        "gpu": gpu_info(),
    }
    write_json(results_dir(cfg, "quantize") / "gptq.json", rec)
    print(rec)


if __name__ == "__main__":
    main()
