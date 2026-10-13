"""Build the comparison table from the raw JSON files under results/<slug>/.

Reads: lm_eval/<variant>/<task>/**/results_*.json, perplexity/<variant>.json,
footprint/<variant>.json. Writes comparison.{md,csv,json} next to them, and can splice the
markdown table into README.md and MODEL_CARD.md between the RESULTS markers. Nothing is estimated: a missing
input shows as n/a, and a variant with missing inputs is INCOMPLETE rather than go/no-go.

Go/no-go rule (thresholds from the config): GO only if
  mean over tasks of (baseline - variant) accuracy <= max_mean_accuracy_drop_pts  AND
  perplexity increase vs baseline <= max_perplexity_increase_pct."""
import csv
import json
import math
from pathlib import Path

from common.config import REPO_ROOT, base_parser, load_config, results_dir

BASELINE = "fp16"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _num(x):
    return float(x) if isinstance(x, (int, float)) else None


def load_task_metric(task_dir, metric, task):
    """Return (value, stderr, n_effective) for the headline metric from the newest results file, or None."""
    files = sorted(Path(task_dir).rglob("results_*.json"))
    if not files:
        return None
    raw = json.load(open(files[-1]))
    res = raw["results"].get(task, {})
    n = raw.get("n-samples", {}).get(task, {}).get("effective")
    if metric not in res:
        raise KeyError(f"{task}: metric '{metric}' not in {sorted(res)} ({files[-1]})")
    stderr_key = metric.replace(",", "_stderr,", 1)
    return _num(res[metric]), _num(res.get(stderr_key)), n


def collect(cfg, variants):
    base = results_dir(cfg)
    tasks = cfg["lm_eval"]["tasks"]
    data = {}
    for v in variants:
        d = {"tasks": {}, "ppl": None, "disk_gb": None, "mem_load_gb": None, "mem_gen_gb": None}
        for t, tc in tasks.items():
            r = load_task_metric(base / "lm_eval" / v / t, tc["headline_metric"], t)
            if r and r[0] is not None:
                d["tasks"][t] = {"acc": 100 * r[0], "se": None if r[1] is None else 100 * r[1], "n": r[2]}
        pf = base / "perplexity" / f"{v}.json"
        if pf.exists():
            d["ppl"] = json.load(open(pf))["perplexity"]
        ff = base / "footprint" / f"{v}.json"
        if ff.exists():
            f = json.load(open(ff))
            d["disk_gb"] = f["weights_bytes_on_disk"] / 1e9
            d["mem_load_gb"] = f["gpu_bytes_after_load"] / 1e9
            d["mem_gen_gb"] = f["gpu_peak_bytes_generate"] / 1e9
        data[v] = d
    return data


def judge(cfg, data, v):
    """Return (verdict, reason, mean_drop_pts, ppl_increase_pct)."""
    th = cfg["thresholds"]
    b, d = data[BASELINE], data[v]
    task_names = list(cfg["lm_eval"]["tasks"])
    missing = [t for t in task_names if t not in d["tasks"] or t not in b["tasks"]]
    if missing or d["ppl"] is None or b["ppl"] is None:
        what = missing + ([] if d["ppl"] is not None and b["ppl"] is not None else ["perplexity"])
        return "INCOMPLETE", f"missing results: {', '.join(what)}", None, None
    drops = [b["tasks"][t]["acc"] - d["tasks"][t]["acc"] for t in task_names]
    mean_drop = sum(drops) / len(drops)
    ppl_inc = 100 * (d["ppl"] / b["ppl"] - 1)
    ok_acc = mean_drop <= th["max_mean_accuracy_drop_pts"]
    ok_ppl = ppl_inc <= th["max_perplexity_increase_pct"]
    if ok_acc and ok_ppl:
        return ("GO", f"mean drop {mean_drop:.1f} pts <= {th['max_mean_accuracy_drop_pts']}, "
                f"perplexity +{ppl_inc:.1f}% <= {th['max_perplexity_increase_pct']}%", mean_drop, ppl_inc)
    why = []
    if not ok_acc:
        why.append(f"mean drop {mean_drop:.1f} pts > {th['max_mean_accuracy_drop_pts']}")
    if not ok_ppl:
        why.append(f"perplexity +{ppl_inc:.1f}% > {th['max_perplexity_increase_pct']}%")
    return "NO-GO", "; ".join(why), mean_drop, ppl_inc


def within_noise(a, b):
    """True if two accuracies differ by less than 2 combined standard errors."""
    if a["se"] is None or b["se"] is None:
        return False
    return abs(a["acc"] - b["acc"]) < 2 * math.hypot(a["se"], b["se"])


def fmt(x, nd=1, suffix=""):
    return "n/a" if x is None else f"{x:.{nd}f}{suffix}"


def build(cfg, data):
    variants = list(data)
    tasks = list(cfg["lm_eval"]["tasks"])
    th = cfg["thresholds"]
    rows, md = [], []

    def n_of(t):  # effective sample count actually evaluated, taken from the baseline's result file
        return (data[BASELINE]["tasks"].get(t) or {}).get("n", "?")

    acc_hdr = [f"{t} (n={n_of(t)})" for t in tasks]
    md.append("| variant | " + " | ".join(acc_hdr) + " | wikitext2 ppl | disk GB | GPU GB (weights) | GPU GB (peak gen) |")
    md.append("|---|" + "---|" * (len(tasks) + 4))
    for v in variants:
        d = data[v]
        cells = []
        for t in tasks:
            a = d["tasks"].get(t)
            if a is None:
                cells.append("n/a")
                continue
            se = f" ± {a['se']:.1f}" if a["se"] is not None else ""
            cells.append(f"{a['acc']:.1f}{se}")
        md.append(f"| {v} | " + " | ".join(cells) + f" | {fmt(d['ppl'], 2)} | {fmt(d['disk_gb'], 2)} | "
                  f"{fmt(d['mem_load_gb'], 2)} | {fmt(d['mem_gen_gb'], 2)} |")
        rows.append({"variant": v, **{t: (d['tasks'].get(t) or {}).get('acc') for t in tasks},
                     "ppl": d["ppl"], "disk_gb": d["disk_gb"], "gpu_gb_weights": d["mem_load_gb"],
                     "gpu_gb_peak_gen": d["mem_gen_gb"]})

    md += ["", "Delta vs fp16 (accuracy points; `~` = within 2 combined standard errors, i.e. not distinguishable at this sample size):", ""]
    md.append("| variant | " + " | ".join(tasks) + " | mean drop (pts) | ppl change |")
    md.append("|---|" + "---|" * (len(tasks) + 2))
    verdicts = {}
    for v in variants:
        if v == BASELINE:
            continue
        cells = []
        for t in tasks:
            a, b = data[v]["tasks"].get(t), data[BASELINE]["tasks"].get(t)
            if a is None or b is None:
                cells.append("n/a")
            else:
                cells.append(f"{a['acc'] - b['acc']:+.1f}" + (" ~" if within_noise(a, b) else ""))
        verdict, reason, mean_drop, ppl_inc = judge(cfg, data, v)
        verdicts[v] = {"verdict": verdict, "reason": reason, "mean_drop_pts": mean_drop, "ppl_increase_pct": ppl_inc}
        md.append(f"| {v} | " + " | ".join(cells) + f" | {fmt(mean_drop)} | " +
                  (f"{ppl_inc:+.1f}%" if ppl_inc is not None else "n/a") + " |")

    md += ["", f"Go/no-go rule: GO only if mean accuracy drop <= {th['max_mean_accuracy_drop_pts']} pts "
           f"across the {len(tasks)} tasks AND wikitext2 perplexity rises <= {th['max_perplexity_increase_pct']}% vs fp16.", "",
           "| variant | call | reason |", "|---|---|---|"]
    for v, j in verdicts.items():
        md.append(f"| {v} | **{j['verdict']}** | {j['reason']} |")
    return "\n".join(md) + "\n", rows, verdicts


def splice_readme(path, table):
    text = Path(path).read_text()
    if START not in text or END not in text:
        raise SystemExit(f"{path} has no {START} / {END} markers")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    Path(path).write_text(f"{head}{START}\n{table}{END}{tail}")


def main():
    p = base_parser(__doc__)
    p.add_argument("--update-readme", action="store_true", help="splice tables into README.md and MODEL_CARD.md")
    p.add_argument("--results-root", help="override results dir (tests only)")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.results_root:
        cfg["paths"]["results_dir"] = args.results_root
    variants = list(cfg["variants"])
    data = collect(cfg, variants)
    table, rows, verdicts = build(cfg, data)

    out = results_dir(cfg)
    (out / "comparison.md").write_text(table)
    with open(out / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "comparison.json").write_text(json.dumps({"rows": rows, "verdicts": verdicts}, indent=2))
    print(table)
    if args.update_readme:
        for name in ("README.md", "MODEL_CARD.md"):
            splice_readme(REPO_ROOT / name, table)


if __name__ == "__main__":
    main()
