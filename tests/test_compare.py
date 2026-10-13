"""Tests for eval/compare.py using SYNTHETIC numbers written to a temp dir.
These values are made up to exercise the logic; they are never written to results/."""
import json

import pytest

from common.config import load_config
from eval import compare

TASKS = {"gsm8k": "exact_match,flexible-extract", "arc_challenge": "acc_norm,none"}


def cfg_for(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg["paths"]["results_dir"] = str(tmp_path)
    cfg["variants"] = {"fp16": {}, "q": {}}
    cfg["lm_eval"]["tasks"] = {t: {"headline_metric": m, "limit": 100} for t, m in TASKS.items()}
    return cfg


def write_variant(tmp_path, cfg, v, accs, ppl, se=0.01):
    base = tmp_path / cfg["model"]["slug"]
    for (t, m), a in zip(TASKS.items(), accs):
        d = base / "lm_eval" / v / t / "sub"
        d.mkdir(parents=True, exist_ok=True)
        key_se = m.replace(",", "_stderr,", 1)
        (d / "results_2000-01-01T00-00-00.json").write_text(
            json.dumps({"results": {t: {m: a, key_se: se}}}))
    if ppl is not None:
        (base / "perplexity").mkdir(parents=True, exist_ok=True)
        (base / "perplexity" / f"{v}.json").write_text(json.dumps({"perplexity": ppl}))


def verdict(tmp_path, cfg):
    data = compare.collect(cfg, list(cfg["variants"]))
    return compare.judge(cfg, data, "q")


def test_go(tmp_path):
    cfg = cfg_for(tmp_path)
    write_variant(tmp_path, cfg, "fp16", [0.60, 0.50], 10.0)
    write_variant(tmp_path, cfg, "q", [0.59, 0.49], 10.5)
    v, reason, drop, inc = verdict(tmp_path, cfg)
    assert v == "GO" and drop == pytest.approx(1.0) and inc == pytest.approx(5.0)


def test_no_go_accuracy(tmp_path):
    cfg = cfg_for(tmp_path)
    write_variant(tmp_path, cfg, "fp16", [0.60, 0.50], 10.0)
    write_variant(tmp_path, cfg, "q", [0.50, 0.45], 10.1)
    v, reason, *_ = verdict(tmp_path, cfg)
    assert v == "NO-GO" and "mean drop" in reason and "perplexity" not in reason


def test_no_go_perplexity(tmp_path):
    cfg = cfg_for(tmp_path)
    write_variant(tmp_path, cfg, "fp16", [0.60, 0.50], 10.0)
    write_variant(tmp_path, cfg, "q", [0.60, 0.50], 12.0)
    v, reason, *_ = verdict(tmp_path, cfg)
    assert v == "NO-GO" and "perplexity" in reason


def test_incomplete_when_data_missing(tmp_path):
    cfg = cfg_for(tmp_path)
    write_variant(tmp_path, cfg, "fp16", [0.60, 0.50], 10.0)
    write_variant(tmp_path, cfg, "q", [0.60, 0.50], None)
    v, reason, *_ = verdict(tmp_path, cfg)
    assert v == "INCOMPLETE" and "perplexity" in reason


def test_noise_marker():
    a, b = {"acc": 60.0, "se": 3.0}, {"acc": 58.0, "se": 3.0}
    assert compare.within_noise(a, b)
    assert not compare.within_noise({"acc": 60.0, "se": 0.5}, {"acc": 50.0, "se": 0.5})
    assert not compare.within_noise({"acc": 60.0, "se": None}, b)


def test_na_stderr_string_is_handled(tmp_path):
    cfg = cfg_for(tmp_path)
    write_variant(tmp_path, cfg, "fp16", [0.6, 0.5], 10.0, se="N/A")
    data = compare.collect(cfg, ["fp16"])
    assert data["fp16"]["tasks"]["gsm8k"]["se"] is None


def test_wrong_metric_key_fails_loudly(tmp_path):
    cfg = cfg_for(tmp_path)
    write_variant(tmp_path, cfg, "fp16", [0.6, 0.5], 10.0)
    cfg["lm_eval"]["tasks"]["gsm8k"]["headline_metric"] = "nope,none"
    with pytest.raises(KeyError):
        compare.collect(cfg, ["fp16"])


def test_full_table_renders_and_splices(tmp_path):
    cfg = cfg_for(tmp_path)
    write_variant(tmp_path, cfg, "fp16", [0.60, 0.50], 10.0)
    write_variant(tmp_path, cfg, "q", [0.59, 0.49], 10.5)
    table, rows, verdicts = compare.build(cfg, compare.collect(cfg, ["fp16", "q"]))
    assert "| q |" in table and verdicts["q"]["verdict"] == "GO"
    readme = tmp_path / "README.md"
    readme.write_text(f"a\n{compare.START}\nold\n{compare.END}\nb\n")
    compare.splice_readme(readme, table)
    out = readme.read_text()
    assert "old" not in out and out.startswith("a\n") and out.endswith("\nb\n")
