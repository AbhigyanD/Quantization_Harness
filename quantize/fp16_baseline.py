"""Download the original FP16 model into models/<slug>/fp16 so every variant is loaded
from a local directory the same way, and record its on-disk size."""
import time

from huggingface_hub import snapshot_download

from common.config import base_parser, hf_token, load_config, model_dir, results_dir
from common.metrics import gpu_info, weights_size_bytes, write_json


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    out = model_dir(cfg, "fp16")
    t0 = time.time()
    snapshot_download(
        cfg["model"]["base"], revision=cfg["model"]["revision"], local_dir=out, token=hf_token()
    )
    rec = {
        "variant": "fp16",
        "base_model": cfg["model"]["base"],
        "revision": cfg["model"]["revision"],
        "download_seconds": round(time.time() - t0, 1),
        "weights_bytes": weights_size_bytes(out),
        "gpu": gpu_info(),
    }
    write_json(results_dir(cfg, "quantize") / "fp16.json", rec)
    print(rec)


if __name__ == "__main__":
    main()
