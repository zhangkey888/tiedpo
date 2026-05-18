# LLaVA-NeXT TieDPO

This directory keeps the maintained Qwen2 / LLaVA-OneVision path only:

- TieDPO training
- A/B-gap evaluation
- lmms-eval benchmark comparison
- custom trainer code used by the maintained recipe

## Required Environment Variables

```bash
export DATA_ROOT=/path/to/data_root
export MODELS_ROOT=/path/to/models_root
export HF_HOME=/path/to/hf_cache
```

Optional:

```bash
export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
export LMMS_EVAL_DIR=/path/to/lmms-eval
```

Keep private API keys, tokens, and any custom model endpoints in your local shell or CI secret store only.

## Training

Run the maintained three-run recipe:

```bash
cd LLaVA-NeXT
bash scripts/train/run_lora_c1_minimal.sh
```

The underlying single-run training entrypoint is:

```bash
cd LLaVA-NeXT
bash scripts/train/TieDPO_all_evidence.sh
```

## A/B-gap Evaluation

```bash
cd LLaVA-NeXT
bash scripts/eval/run_qwen2_7b_3model_abgap_eval.sh
```

## Benchmark Evaluation

```bash
cd LLaVA-NeXT
LMMS_EVAL_DIR=/path/to/lmms-eval \
bash scripts/eval/run_3model_lmms_eval_and_summarize.sh
```

## Core Code

- `llava/train/train_tie_dpo.py`
- `llava/train/llava_trainer.py`
- `trl/trainer/tie_dpo_trainer.py`
- `data_processing/utils.py`

## Notes

- Local checkpoints, datasets, evaluation outputs, and merged models are ignored by the repo-level `.gitignore`.
- The kept shell scripts are the maintained training / evaluation entrypoints. Old one-off environment bootstrap scripts were intentionally removed from the curated repo.
