"""Perplexity of one variant on WikiText-2 (test), plain PyTorch.

Standard sliding-window protocol: windows of max_length tokens advance by `stride`; only
tokens not already scored by an earlier window contribute to the loss, so every token is
scored once with at least (max_length - stride) tokens of context (except the first window).
Tokenization is of the whole split joined with blank lines, as in the HF perplexity guide."""
import math
import time

import torch
from datasets import load_dataset

from common.config import base_parser, load_config, results_dir, set_seed
from common.metrics import gpu_info, write_json
from common.models import load_variant


@torch.no_grad()
def compute_perplexity(model, input_ids, max_length, stride, device):
    n = input_ids.size(1)
    nll_sum, n_scored, prev_end = 0.0, 0, 0
    for begin in range(0, n, stride):
        end = min(begin + max_length, n)
        trg_len = end - prev_end  # tokens in this window not yet scored
        ids = input_ids[:, begin:end].to(device)
        labels = ids.clone()
        labels[:, :-trg_len] = -100
        logits = model(ids).logits.float()
        # shift: position i predicts token i+1
        loss = torch.nn.functional.cross_entropy(
            logits[0, :-1], labels[0, 1:], ignore_index=-100, reduction="sum"
        )
        n_tok = int((labels[0, 1:] != -100).sum())
        nll_sum += loss.item()
        n_scored += n_tok
        prev_end = end
        if end == n:
            break
    return math.exp(nll_sum / n_scored), n_scored


def main():
    p = base_parser(__doc__)
    p.add_argument("--variant", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", help="override output path (smoke tests only)")
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    pc = cfg["perplexity"]

    model, tok = load_variant(cfg, args.variant, device=args.device)
    ds = load_dataset(pc["dataset"], pc["config"], split=pc["split"])
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    if pc.get("max_tokens"):
        ids = ids[:, : pc["max_tokens"]]

    t0 = time.time()
    ppl, n_scored = compute_perplexity(model, ids, pc["max_length"], pc["stride"], args.device)
    rec = {
        "variant": args.variant,
        "perplexity": ppl,
        "tokens_scored": n_scored,
        "tokens_total": ids.size(1),
        "settings": pc,
        "seconds": round(time.time() - t0, 1),
        "gpu": gpu_info(),
    }
    out = args.out or results_dir(cfg, "perplexity") / f"{args.variant}.json"
    write_json(out, rec)
    print(rec)


if __name__ == "__main__":
    main()
