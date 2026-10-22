# P-Check multilingual test (free version)

This experiment tests whether P-Check (Seo & Lee, ACL 2026), a personalized LLM judge, gets less accurate when a user's past conversations are in Korean, Chinese, Malay or a mix of languages instead of English. It needs no training and runs on Kaggle's free GPUs.

## Files

| File | What it is |
| --- | --- |
| `pcheck_multilingual_kaggle.ipynb` | The notebook. Upload it to Kaggle and run it. |
| `pcheck_ml.py` | The same pipeline code the notebook writes out. It's here so you can read it more easily. |

## How to run

1. Go to kaggle.com, then *Create → New Notebook → File → Import Notebook*, and pick the `.ipynb` file.
2. Under *Settings*, set *Accelerator* to **GPU T4 x2** and turn *Internet* on.
3. On Hugging Face, accept the Llama 3.2 3B Instruct and Llama 3.1 8B Instruct licenses. Make a read token.
4. In Kaggle, open *Add-ons → Secrets* and add the token with the name `HF_TOKEN`.
5. Click *Save Version → Save & Run All*. It takes roughly 3–5 hours, and you can close the browser while it runs.
6. When it finishes, open the version's *Output* tab and download `out/results.md` and `out/handcheck.csv`.

A 5-user smoke test runs first, so a setup problem shows up in about 20 minutes. If any stage fails, the error is in the log files next to `out/`. Send the error to Claude.

## Design

- **Users:** 100 PRISM test users, picked at random with seed 42. That gives 377 chosen-vs-rejected test pairs and 1,039 past conversations.
- **History versions:**
  - English (the original)
  - Korean, Chinese and Malay, each translated with NLLB-200 (600M distilled)
  - Mixed: past conversation 1 is in English, 2 in Chinese, 3 in Malay, 4 in Korean, and so on. This is mixing between conversations. It is not code-switching inside a sentence.
- **What stays in English:** the test question and both candidate answers. So any change in accuracy comes from the history language alone.
- **Same content in every version:** if a long history doesn't fit in the 16K-token context, the oldest conversations are dropped. The same ones are dropped in every language version.
- **Profile (GP):** the paper uses GPT-4o-mini. Here Llama-3.1-8B-Instruct (4-bit AWQ) writes it, using the repo's own prompt.
- **Checklist:** the paper uses its *trained* Llama-3.2-3B. Here the *base* model is used, with the repo's prompt plus a note on the JSON output format. Don't compare your accuracy with the paper's; compare the languages with each other.
- **Judge:** Llama-3.1-8B-Instruct (4-bit). It runs in two settings:
  - *Plain*: one 1–10 score per answer from the profile. This prompt is our own.
  - *Checklist*: the repo's per-criterion scoring prompt, with the repo's weights (Essential 1.0, Important 0.7, Optional 0.3).
- **Settings:** greedy decoding (temperature 0) so runs repeat. The repo uses 1.0.
- **Metric:** a pair counts as correct when the chosen answer scores higher than the rejected one. Ties count as wrong, as in the repo's `eval.py`. Each accuracy comes with a 95% bootstrap confidence interval, and each language also gets a paired difference against English.
- **How small a drop it can detect:** with about 377 pairs, a difference against English needs to be about 7 points or more to clearly stand out from noise.

## What was tested before delivery

Tested without a GPU:
- History parsing and rebuilding. The rebuilt prompts match the original GP prompts exactly for all 362 users.
- Translation segmenting and rejoining, sharding and resuming, and building every prompt.
- Scoring and the results table, using fake model outputs.

Not tested, because no GPU was available while this was written:
- The vLLM install on Kaggle
- Model downloads
- NLLB speed and translation quality
- Real run times

The smoke test is there to catch these early.
