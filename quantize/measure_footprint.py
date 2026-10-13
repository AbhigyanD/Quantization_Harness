"""Peak GPU memory for one variant, in a fresh process (run once per variant so the
peak counter isn't polluted by another model). Two numbers:
  load_peak  = GPU memory after loading weights (what the model costs to hold)
  gen_peak   = peak during a fixed generation (prompt_tokens in, new_tokens out)."""
import torch

from common.config import base_parser, load_config, results_dir, set_seed
from common.metrics import gpu_info, weights_size_bytes, write_json
from common.models import load_variant, variant_path


def main():
    p = base_parser(__doc__)
    p.add_argument("--variant", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    f = cfg["footprint"]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tok = load_variant(cfg, args.variant)
    torch.cuda.synchronize()
    load_peak = torch.cuda.max_memory_allocated()
    weights_resident = torch.cuda.memory_allocated()

    # Fixed-length dummy prompt: same token ids for every variant (same tokenizer).
    ids = torch.full((f["batch_size"], f["prompt_tokens"]), tok.encode("the")[0], device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=f["new_tokens"],
            min_new_tokens=f["new_tokens"],
            do_sample=False,
        )
    torch.cuda.synchronize()
    gen_peak = torch.cuda.max_memory_allocated()

    rec = {
        "variant": args.variant,
        "weights_bytes_on_disk": weights_size_bytes(variant_path(cfg, args.variant)),
        "gpu_bytes_after_load": weights_resident,
        "gpu_peak_bytes_load": load_peak,
        "gpu_peak_bytes_generate": gen_peak,
        "workload": f,
        "gpu": gpu_info(),
    }
    write_json(results_dir(cfg, "footprint") / f"{args.variant}.json", rec)
    print(rec)


if __name__ == "__main__":
    main()
