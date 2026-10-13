"""Turn results/<slug>/bench/<variant>.json into a markdown table (and optionally splice it
into README.md between the BENCH markers). Variants that did not start are listed with the
server's own error line, not dropped."""
import json
from pathlib import Path

from common.config import REPO_ROOT, base_parser, load_config, results_dir

START, END = "<!-- BENCH:START -->", "<!-- BENCH:END -->"


def _f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def last_error_line(tail):
    for line in reversed(tail or []):
        if any(k in line for k in ("Error", "error", "ERROR", "Exception")):
            return line.strip()[:300]
    return (tail or ["(no log)"])[-1].strip()[:300]


def build_table(recs):
    md = ["| variant | concurrency | output tok/s | TTFT p50 (s) | TTFT p95 (s) | latency p50 (s) | latency p95 (s) | failed |",
          "|---|---|---|---|---|---|---|---|"]
    broken = []
    for r in recs:
        if r.get("failed_to_start") or not r.get("levels"):
            broken.append(r)
            continue
        for l in r["levels"]:
            t, lat = l["ttft_s"] or {}, l["latency_s"] or {}
            md.append(f"| {r['variant']} | {l['concurrency']} | {_f(l['output_tokens_per_s'], 1)} | "
                      f"{_f(t.get('p50'), 3)} | {_f(t.get('p95'), 3)} | {_f(lat.get('p50'))} | "
                      f"{_f(lat.get('p95'))} | {l['failed']}/{l['requests']} |")
    out = "\n".join(md) + "\n"
    if broken:
        out += "\nVariants vLLM did not serve:\n\n"
        for r in broken:
            out += f"- **{r['variant']}**: {r.get('server_status', 'no result')}. Last relevant log line: `{last_error_line(r.get('server_log_tail'))}`\n"
    return out


def main():
    p = base_parser(__doc__)
    p.add_argument("--update-readme", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    d = results_dir(cfg, "bench")
    recs = [json.load(open(f)) for f in sorted(Path(d).glob("*.json"))]
    table = build_table(recs)
    (d.parent / "bench_summary.md").write_text(table)
    print(table)
    if args.update_readme:
        for name in ("README.md", "MODEL_CARD.md"):
            path = REPO_ROOT / name
            head, rest = path.read_text().split(START, 1)
            _, tail = rest.split(END, 1)
            path.write_text(f"{head}{START}\n{table}{END}{tail}")


if __name__ == "__main__":
    main()
