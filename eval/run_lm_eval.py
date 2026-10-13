"""Run the configured lm-evaluation-harness tasks for one variant.

Each task is a separate `lm_eval run` call so that per-task limit / few-shot settings
are explicit. Settings are identical across variants; only --variant changes. Raw lm_eval
JSON goes to results/<slug>/lm_eval/<variant>/<task>/."""
import subprocess
import sys

from common.config import base_parser, load_config, results_dir
from common.metrics import gpu_info, pip_freeze, write_json
from common.models import variant_path

GEN_TASKS = {"gsm8k", "ifeval"}
NLTK_TASKS = {"ifeval", "niah_multikey_1"}


def ensure_nltk():
    """ifeval and RULER need nltk's punkt_tab. Fail early with a clear message instead of
    deep inside a task (a missing tokenizer was a silent failure mode, see engineering log)."""
    import nltk
    if not nltk.download("punkt_tab", quiet=True) or not _has_punkt(nltk):
        sys.exit("nltk punkt_tab is unavailable (download failed, check SSL certs / network)")


def _has_punkt(nltk):
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
        return True
    except LookupError:
        return False


def build_cmd(cfg, variant, task, tcfg, out_dir):
    seed = cfg["seed"]
    margs = f"pretrained={variant_path(cfg, variant)},dtype=float16"
    if "max_length" in tcfg:
        margs += f",max_length={tcfg['max_length']}"
    cmd = [
        sys.executable, "-m", "lm_eval", "run",
        "--model", "hf",
        "--model_args", margs,
        "--tasks", task,
        "--num_fewshot", str(tcfg["num_fewshot"]),
        "--batch_size", str(cfg["lm_eval"]["batch_size"]),
        "--seed", f"{seed},{seed},{seed},{seed}",
        "--output_path", str(out_dir),
    ]
    if tcfg.get("limit") is not None:
        cmd += ["--limit", str(tcfg["limit"])]
    if cfg["lm_eval"]["apply_chat_template"]:
        cmd += ["--apply_chat_template"]
        if tcfg.get("fewshot_as_multiturn"):
            cmd += ["--fewshot_as_multiturn"]
    if task in GEN_TASKS:
        cmd += ["--gen_kwargs", f"max_gen_toks={cfg['lm_eval']['max_gen_toks']}"]
    if "metadata" in tcfg:
        cmd += ["--metadata", tcfg["metadata"]]
    return cmd


def main():
    p = base_parser(__doc__)
    p.add_argument("--variant", required=True)
    p.add_argument("--tasks", nargs="*", help="subset of configured tasks (default: all)")
    p.add_argument("--limit", type=int, help="override every task's limit (smoke tests only)")
    p.add_argument("--out-dir", help="override output dir (smoke tests only)")
    args = p.parse_args()
    cfg = load_config(args.config)

    tasks = cfg["lm_eval"]["tasks"]
    names = args.tasks or list(tasks)
    if NLTK_TASKS & set(names):
        ensure_nltk()
    base_out = args.out_dir or results_dir(cfg, "lm_eval", args.variant)

    run_log = {"variant": args.variant, "gpu": gpu_info(), "commands": []}
    failed = []
    for name in names:
        tcfg = dict(tasks[name])
        if args.limit is not None:
            tcfg["limit"] = args.limit
        cmd = build_cmd(cfg, args.variant, name, tcfg, f"{base_out}/{name}")
        print("+", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd).returncode
        run_log["commands"].append({"task": name, "cmd": cmd, "returncode": rc})
        if rc != 0:
            failed.append(name)

    run_log["pip_freeze"] = pip_freeze()
    write_json(f"{base_out}/run_log.json", run_log)
    if failed:
        sys.exit(f"lm_eval failed for: {failed}")


if __name__ == "__main__":
    main()
