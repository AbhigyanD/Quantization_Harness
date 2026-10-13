"""vLLM serving benchmark for one or more variants.

For each variant: start `vllm serve` on the variant's checkpoint, wait until healthy, then
for each concurrency level send `requests_per_level` streaming completions (closed loop:
at most `concurrency` in flight) and record
  TTFT            time from sending the request to the first streamed text chunk
  latency         time from sending the request to the end of the stream
  TPOT            (latency - TTFT) / (output_tokens - 1)
  output tok/s    total generated tokens / wall time of the level
Prompts are distinct wikitext slices of fixed token length (so prefix caching can't make
requests free); output length is fixed with ignore_eos so every request does equal work.
If the server fails to start, the failure reason and the log tail go into the JSON instead
of the run silently skipping the variant."""
import asyncio
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time

import aiohttp

from common.config import base_parser, load_config, results_dir, set_seed
from common.metrics import gpu_info, write_json
from common.models import variant_path


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summarize(xs):
    if not xs:
        return None
    return {"mean": statistics.mean(xs), "p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "max": max(xs)}


async def one_request(session, url, model, prompt, max_tokens):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft, out_tokens = None, None
    try:
        async with session.post(url, json=body) as resp:
            if resp.status != 200:
                return {"error": f"HTTP {resp.status}: {(await resp.text())[:300]}"}
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("usage"):
                    out_tokens = chunk["usage"]["completion_tokens"]
                if ttft is None and chunk.get("choices") and chunk["choices"][0].get("text"):
                    ttft = time.perf_counter() - t0
    except Exception as e:  # connection reset, server crash mid-run, etc.
        return {"error": f"{type(e).__name__}: {e}"}
    latency = time.perf_counter() - t0
    if ttft is None or not out_tokens:
        return {"error": "no tokens streamed"}
    tpot = (latency - ttft) / (out_tokens - 1) if out_tokens > 1 else None
    return {"ttft": ttft, "latency": latency, "out_tokens": out_tokens, "tpot": tpot}


async def run_level(base_url, model, prompts, concurrency, max_tokens):
    url = f"{base_url}/v1/completions"
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as s:
        async def guarded(p):
            async with sem:
                return await one_request(s, url, model, p, max_tokens)
        t0 = time.perf_counter()
        results = await asyncio.gather(*(guarded(p) for p in prompts))
        wall = time.perf_counter() - t0
    ok = [r for r in results if "error" not in r]
    errors = [r["error"] for r in results if "error" in r]
    return {
        "concurrency": concurrency,
        "requests": len(prompts),
        "succeeded": len(ok),
        "failed": len(errors),
        "first_errors": errors[:3],
        "wall_seconds": wall,
        "output_tokens_per_s": sum(r["out_tokens"] for r in ok) / wall if ok else None,
        "requests_per_s": len(ok) / wall if ok else None,
        "ttft_s": summarize([r["ttft"] for r in ok]),
        "latency_s": summarize([r["latency"] for r in ok]),
        "tpot_s": summarize([r["tpot"] for r in ok if r["tpot"] is not None]),
    }


def build_prompts(cfg, tokenizer, n, tokens, seed):
    from datasets import load_dataset
    pc = cfg["perplexity"]
    ds = load_dataset(pc["dataset"], pc["config"], split="train")
    ids = tokenizer("\n\n".join(t for t in ds["text"] if t.strip()), add_special_tokens=False).input_ids
    rng = random.Random(seed)
    starts = rng.sample(range(0, len(ids) - tokens - 1), n)
    return [tokenizer.decode(ids[s : s + tokens]) for s in starts]


def tail(path, n=40):
    try:
        return open(path, errors="replace").read().splitlines()[-n:]
    except OSError:
        return []


def start_server(cfg, variant, log_path):
    b = cfg["bench"]
    sibling = os.path.join(os.path.dirname(sys.executable), "vllm")  # vLLM lives in its own venv
    cmd = [sibling if os.path.exists(sibling) else "vllm", "serve", variant_path(cfg, variant),
           "--served-model-name", variant, "--dtype", b["dtype"],
           "--max-model-len", str(b["max_model_len"]),
           "--gpu-memory-utilization", str(b["gpu_memory_utilization"]),
           "--port", str(b["port"]), "--seed", str(cfg["seed"])]
    cmd += cfg["variants"][variant].get("vllm_args", [])
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return proc, cmd


def stop_server(proc):
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
    time.sleep(5)  # let the GPU memory actually free before the next variant


def wait_healthy(proc, port, timeout_s):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return False, f"server exited with code {proc.returncode}"
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True, f"ready after {time.time() - t0:.0f}s"
        except Exception:
            pass
        time.sleep(3)
    return False, f"not healthy after {timeout_s}s"


def bench_variant(cfg, variant, prompts_by_level, warmup_prompts):
    b = cfg["bench"]
    out_dir = results_dir(cfg, "bench")
    log_path = out_dir / f"server_{variant}.log"
    rec = {"variant": variant, "settings": b, "gpu": gpu_info()}
    try:
        import vllm
        rec["vllm_version"] = vllm.__version__
    except Exception:
        rec["vllm_version"] = None
    proc, cmd = start_server(cfg, variant, log_path)
    rec["server_cmd"] = cmd
    try:
        ok, msg = wait_healthy(proc, b["port"], b["server_start_timeout_s"])
        rec["server_status"] = msg
        if not ok:
            rec["failed_to_start"] = True
            rec["server_log_tail"] = tail(log_path)
            return rec
        base_url = f"http://127.0.0.1:{b['port']}"
        asyncio.run(run_level(base_url, variant, warmup_prompts, 1, b["output_tokens"]))
        rec["levels"] = []
        for c in b["concurrency"]:
            lvl = asyncio.run(run_level(base_url, variant, prompts_by_level[c], c, b["output_tokens"]))
            print(f"[{variant}] c={c}: {lvl['output_tokens_per_s']} tok/s, failed={lvl['failed']}", flush=True)
            rec["levels"].append(lvl)
            if lvl["failed"] and proc.poll() is not None:
                rec["server_log_tail"] = tail(log_path)
                break
    finally:
        stop_server(proc)
    return rec


def main():
    p = base_parser(__doc__)
    p.add_argument("--variants", nargs="*", help="default: all variants in the config")
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    b = cfg["bench"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(variant_path(cfg, "fp16"))
    n_total = b["warmup_requests"] + sum(b["requests_per_level"] for _ in b["concurrency"])
    prompts = build_prompts(cfg, tok, n_total, b["input_tokens"], cfg["seed"])
    warm, rest = prompts[: b["warmup_requests"]], prompts[b["warmup_requests"]:]
    by_level = {c: rest[i * b["requests_per_level"]:(i + 1) * b["requests_per_level"]]
                for i, c in enumerate(b["concurrency"])}

    for v in args.variants or list(cfg["variants"]):
        rec = bench_variant(cfg, v, by_level, warm)
        write_json(results_dir(cfg, "bench") / f"{v}.json", rec)
        print(f"[{v}] {rec['server_status']}", flush=True)


if __name__ == "__main__":
    main()
