# Engineering log

What broke, how I found out, and what fixed it. Entries are added as they happen.
Entries marked **(local)** came from developing the scripts on a Mac without a GPU;
GPU-run entries are added after the Colab/Kaggle runs.

## Decision: GPTQ instead of AWQ
- **Context:** the task allowed AWQ or GPTQ, whichever ran cleanly on the GPU.
- **What I found:** the AutoAWQ repo is archived and its README says it is deprecated; the last
  tested versions it lists are torch 2.6.0 and transformers 4.51.3. GPTQModel is maintained and
  lists Turing (sm_75, which is the T4) as supported.
- **Choice:** quantize with GPTQModel and drop the AWQ script rather than ship a path I could not
  verify. If a later run shows GPTQModel failing on the T4, the fallback is llm-compressor's AWQ.

## `pip freeze` helper called a `python` that does not exist **(local)**
- **Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'python'` at the end of
  `eval/run_lm_eval.py`, after all the tasks had already run.
- **Cause:** `common/metrics.pip_freeze` shelled out to `python`; on this machine only `python3`
  (and the venv's interpreter) exist.
- **Fix:** use `sys.executable`, so the freeze comes from the same interpreter that ran the eval.

## lm_eval failed with `No module named 'accelerate'` **(local)**
- **Cause:** I had installed packages by hand into a fresh venv instead of from `requirements.txt`.
  `accelerate` is in the requirements file and lm_eval's HF backend needs it.
- **Fix:** install from `requirements.txt` (`requirements-dev.txt` for local work).

## RULER long-context task needs extra packages and NLTK data **(local)**
- **Error 1:** `ImportError: Please install the wonderwords and nltk packages` when running
  `niah_multikey_1`. **Fix:** require `lm_eval[ifeval,ruler]` in `requirements.txt`.
- **Error 2:** `LookupError: Resource 'punkt_tab' not found`. The log also had
  `[nltk_data] Error loading punkt_tab: CERTIFICATE_VERIFY_FAILED`. **Cause:** this Python install
  has no CA bundle that `urllib` can use, so `nltk.download` failed. IFEval printed the same SSL
  error but then continued, so a missing tokenizer can fail quietly inside a task.
- **Fix:** `eval/run_lm_eval.py` now downloads and checks `punkt_tab` before starting and exits with
  a clear message if it is missing. Locally I set `SSL_CERT_FILE` to the `certifi` bundle.
- **Note:** RULER also downloads its haystack essays from GitHub at run time, so the eval needs
  internet access.

## `load_dataset("wikitext", ...)` fails with datasets 5.x **(local)**
- **Error:** `HfUriError: Repository id must be 'namespace/name', got 'wikitext'` from
  `eval/perplexity.py`.
- **Cause:** the short dataset name is no longer resolved by the version of `datasets` that pip
  installed (5.1.0 at the time).
- **Fix:** use `Salesforce/wikitext` in the perplexity and calibration config.
