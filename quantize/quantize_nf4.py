"""bitsandbytes NF4 (double quant) quantization. Weights are quantized as they are
loaded onto the GPU; the result is saved so it can be loaded from disk like the others."""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common.config import base_parser, load_config, model_dir, results_dir, set_seed
from common.metrics import gpu_info, weights_size_bytes, write_json


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    v = cfg["variants"]["nf4"]
    src, out = model_dir(cfg, "fp16"), model_dir(cfg, "nf4")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=v["double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, v["compute_dtype"]),
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(src, quantization_config=bnb, device_map="cuda")
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated()

    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(src).save_pretrained(out)

    rec = {
        "variant": "nf4",
        "config": v,
        "quantize_seconds": round(elapsed, 1),
        "peak_gpu_bytes_during_quantization": peak,
        "weights_bytes": weights_size_bytes(out),
        "gpu": gpu_info(),
    }
    write_json(results_dir(cfg, "quantize") / "nf4.json", rec)
    print(rec)


if __name__ == "__main__":
    main()
