# Quantization_Harness

A harness I built to measure what 4-bit quantization costs a small LLM. It takes
Qwen2.5-1.5B-Instruct, makes two 4-bit versions (bitsandbytes NF4 and GPTQ), and compares both
to the FP16 original on accuracy, perplexity, disk size, GPU memory, and vLLM serving speed.
Everything is driven by `configs/default.yaml` and the scripts in this repo, and the sample and batch
sizes were chosen with a free Colab or Kaggle T4 (16 GB) in mind. Whether they actually fit has not
been confirmed yet; see Status.

## Status

**The scripts are written and the pieces that run without a GPU are tested. The GPU runs have
not been executed yet, so there are no results in this repo.** The tables below stay empty until
`scripts/run_pipeline.sh` and `scripts/run_bench.sh` have run on a T4 and their raw JSON is
committed under `results/`. I do not fill them by hand, and I do not quote numbers from anywhere
else.

What has been checked so far, on a Mac with no GPU:
- `lm_eval` runner, perplexity script and compare script ran end to end on Qwen2.5-0.5B on CPU with
  3 samples per task (smoke test only, those numbers are not kept).
- The compare logic and the vLLM benchmark client have unit tests (`pytest tests`). The client test
  uses a fake streaming server with known timings.
- Not yet run at all: GPTQ quantization, NF4 quantization, GPU memory measurement, anything
  involving vLLM itself.

## What is compared

| Variant | How it is made |
|---|---|
| `fp16` | The original `Qwen/Qwen2.5-1.5B-Instruct`, float16. The baseline for every comparison. |
| `nf4` | bitsandbytes 4-bit NormalFloat with double quantization, fp16 compute. Quantized at load time, then saved. |
| `gptq` | GPTQModel, 4-bit, group size 128, symmetric, `desc_act=false`. Calibrated on 128 random 512-token chunks of the WikiText-2 **train** split. |

I started out planning to include AWQ. I dropped it because the AutoAWQ project is archived and
deprecated; see `docs/engineering_log.md`.

## Settings (same for every variant)

| Item | Setting |
|---|---|
| GPU | one free-tier T4 (16 GB), fp16 only (no bf16 on T4) |
| Seed | 1234 (python, numpy, torch, few-shot sampling) |
| Backend for lm-eval | `hf`, batch size 8, greedy decoding, chat template applied, `max_gen_toks=512` |
| GSM8K | first 250 test problems, 5-shot (as multi-turn), metric: flexible-extract exact match |
| ARC-Challenge | first 500 test questions, 0-shot, metric: `acc_norm` |
| IFEval | first 250 prompts, metric: prompt-level strict accuracy |
| Long context | RULER `niah_multikey_1` at 4096 tokens, 100 generated samples, accuracy |
| Perplexity | WikiText-2 test, whole split, window 2048, stride 1024 (own PyTorch code in `eval/perplexity.py`) |
| Memory | measured in a fresh process per variant: GPU memory after load, and peak during a fixed generation (batch 1, 512 prompt tokens, 128 new tokens) |
| vLLM | `max_model_len` 4096, 256 input / 128 output tokens, concurrency levels 1, 4, 16 and 32, 64 requests per level |

`--limit N` in lm-eval takes the first N documents, so the subsets are the same for every variant
but are not random samples. The sample sizes are small to fit a free GPU session. The tables show
standard errors, and the delta table marks differences smaller than two combined standard errors
with `~`. With n=250 the standard error on an accuracy near 50% is about 3 points, so small
deltas are not distinguishable from noise.

## Go / no-go rule

A quantized variant is **GO** only if both hold:
1. The mean accuracy drop vs `fp16`, averaged over the four tasks, is at most **2.0 points**.
2. WikiText-2 perplexity rises by at most **10%** vs `fp16`.

Otherwise it is **NO-GO**, and the reason line says which condition failed. If any input is
missing, it is **INCOMPLETE**. The thresholds are in `configs/default.yaml` (`thresholds:`) and are
my choice, not a standard. The rule is a screening threshold, not a statistical test: given the
sample sizes, a variant can pass by landing inside the noise.

## Results

<!-- RESULTS:START -->
_No results yet. Run the pipeline on a GPU, then `python -m eval.compare --update-readme`._
<!-- RESULTS:END -->

## vLLM serving benchmark

<!-- BENCH:START -->
_No results yet. Run `scripts/run_bench.sh` on a GPU._
<!-- BENCH:END -->

Notes on how to read it: concurrency is the number of requests in flight against the server
(vLLM batches continuously), not a fixed batch size. Output length is fixed with `ignore_eos`, so
every request does the same amount of work. Prompts are distinct WikiText slices so prefix caching
cannot make requests free. Where vLLM could not serve a variant, the table says so and quotes the
server's error.

## Reproduce

On Colab or Kaggle, open `notebooks/run_on_gpu.ipynb` with a T4 runtime. It clones this repo,
installs dependencies and calls the scripts below; nothing else lives in the notebook. To run by
hand on a GPU machine:

```bash
git clone https://github.com/AbhigyanD/Quantization_Harness.git && cd Quantization_Harness
pip install -r requirements.txt -r requirements-quant.txt
bash scripts/run_pipeline.sh            # download fp16, make nf4 + gptq, memory, perplexity, lm-eval, table
bash scripts/run_bench.sh               # vLLM in its own venv (it pins a different torch)
python -m eval.compare --update-readme  # rebuild the results table in README and MODEL_CARD
python -m bench.summarize --update-readme
```

Individual steps: `python -m quantize.quantize_gptq`, `python -m eval.perplexity --variant gptq`,
`python -m eval.run_lm_eval --variant gptq --tasks gsm8k`, and so on. All take `--config`.
Locally, `pip install -r requirements-dev.txt && pytest tests` runs the tests without a GPU.
For gated models (Llama) set `HF_TOKEN`; see `.env.example`. Nothing here needs any other API key.

To switch to the fallback model, set `model.base` to `meta-llama/Llama-3.2-1B-Instruct` and
`model.slug` to something new in the config; results go under `results/<slug>/`.

## Layout

| Path | Purpose |
|---|---|
| `configs/default.yaml` | All settings: model, seed, sample sizes, thresholds, vLLM levels |
| `quantize/` | FP16 fetch, NF4 and GPTQ scripts, GPU memory measurement |
| `eval/` | lm-eval runner, perplexity, comparison table and go/no-go |
| `bench/` | vLLM serving benchmark and its summary table |
| `common/` | Config, model loading, size helpers |
| `results/<slug>/` | Raw JSON from every run (lm-eval output, perplexity, footprint, bench, run logs with `pip freeze`) |
| `docs/engineering_log.md` | What broke and how I fixed it |
| `MODEL_CARD.md` | Hugging Face model card |
| `notebooks/` | The thin Colab/Kaggle notebook |

## Limitations

- **Small samples.** 250 / 500 / 250 / 100 items per task, first-N subsets, not random draws.
  Differences of a couple of points are within noise. A full-set run would take more GPU time than
  a free session gives.
- **One model, one GPU type, one seed.** Results are for Qwen2.5-1.5B-Instruct on a T4. They say
  little about larger models or other GPUs, and I did not measure run-to-run variation.
- **GPTQ calibration and perplexity both use WikiText-2** (train vs test split). That favors GPTQ's
  perplexity relative to its behaviour on other text. NF4 needs no calibration data.
- **Evaluation goes through the Hugging Face backend, not vLLM.** So accuracy reflects the weights
  and the HF kernels; vLLM's kernels for the same checkpoint could differ slightly.
- **Serving numbers are synthetic.** Fixed-length prompts, fixed-length outputs, closed-loop load,
  one server per variant, 64 requests per level. Real traffic will look different.
- **Long context is one synthetic task at 4096 tokens**, well under the model's 32k limit, because
  of T4 memory and session time.
- **The go/no-go thresholds are my own choice**, not an established standard.
- **Hyperparameters were not tuned.** GPTQ group size, calibration size, etc. are single settings.
