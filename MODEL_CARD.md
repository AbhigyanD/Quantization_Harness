---
base_model: Qwen/Qwen2.5-1.5B-Instruct
license: apache-2.0
tags:
  - quantization
  - gptq
  - bitsandbytes
  - nf4
---

# Qwen2.5-1.5B-Instruct, 4-bit quantized variants

**Status: draft.** I have not run the GPU evaluation yet, so this card has no results and no
weights have been uploaded. The results sections below are filled in by script from files in
`results/` after a real run. Until then they say so.

## What this is

Two 4-bit versions of [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
that I made to measure what 4-bit quantization costs on a small model, plus the original FP16
model as the baseline. The code that makes and tests them is in the repo this card lives in.
I did not train or fine-tune anything. The base model is Apache-2.0, and these variants inherit
that license.

| Variant | Method |
|---|---|
| `fp16` | Original weights, float16 |
| `nf4` | bitsandbytes NF4 with double quantization, fp16 compute |
| `gptq` | GPTQModel, 4-bit, group size 128, symmetric, `desc_act=false`, 128 calibration chunks of 512 tokens from WikiText-2 train |

## How I evaluated them

All variants were loaded the same way and evaluated with identical settings on one free-tier T4:

- lm-evaluation-harness, Hugging Face backend, greedy decoding, chat template, seed 1234
- GSM8K (first 250 problems, 5-shot), ARC-Challenge (first 500), IFEval (first 250 prompts),
  RULER multi-key needle-in-a-haystack at 4096 tokens (100 generated samples)
- Perplexity on the WikiText-2 test split (window 2048, stride 1024), computed with my own PyTorch
  script, not lm-eval
- Disk size and GPU memory measured per variant, in a fresh process each

The exact commands and configuration are in `configs/default.yaml` and `scripts/`. The raw JSON for
every number is under `results/`.

## Results

<!-- RESULTS:START -->
_No results yet._
<!-- RESULTS:END -->

How to read the accuracy columns: the sample sizes are small, so each accuracy has a standard error
of roughly 2 to 5 points. Differences marked `~` are smaller than two combined standard errors and
should be read as "no detectable difference", not as "the same".

My go/no-go rule is a screening threshold I picked: a variant is GO if the mean accuracy drop vs
FP16 is at most 2.0 points and perplexity rises at most 10%. It is not a statistical test.

## Serving with vLLM

<!-- BENCH:START -->
_No results yet._
<!-- BENCH:END -->

## Intended use and limits

These are experiment artifacts for studying quantization, not a product. They inherit the base
model's behavior, including its mistakes and biases, and I did not run any safety evaluation. A
variant that passes my threshold has only been shown not to lose much on the four tasks and the
perplexity set above; it may behave differently on other tasks, languages or long contexts beyond
4096 tokens.

Other limits: one model, one GPU type, one seed; first-N subsets rather than random samples;
GPTQ was calibrated on WikiText-2 and perplexity is also measured on WikiText-2 (different splits,
same domain), which flatters GPTQ's perplexity; accuracy was measured through the Hugging Face
backend, not through vLLM.

## Reproduce

See the README in the repository. In short: `bash scripts/run_pipeline.sh`, then
`bash scripts/run_bench.sh`, then `python -m eval.compare --update-readme`.
