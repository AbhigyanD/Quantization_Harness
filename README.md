# Quantization_Harness

A reproducible harness for comparing 4-bit quantized versions of a small open-weight LLM
(default: Qwen2.5-1.5B-Instruct) against the FP16 original: accuracy, perplexity, memory,
disk size, and vLLM serving latency/throughput.

**Status: work in progress.** The scripts are written; no GPU results have been produced yet.
This README will only contain numbers that come from runs saved under `results/`.

## Layout

| Path | Purpose |
|---|---|
| `configs/` | All settings (model, seeds, sample sizes, thresholds, vLLM batch sizes) |
| `quantize/` | GPTQ / AWQ / bitsandbytes NF4 scripts, footprint measurement |
| `eval/` | lm-evaluation-harness runner, perplexity script, comparison table generator |
| `bench/` | vLLM serving benchmark |
| `results/` | Raw JSON outputs from every run |
| `docs/` | Engineering log |
| `notebooks/` | Thin Colab/Kaggle notebooks (clone, install, call scripts) |
