# TieDPO

This repository is trimmed to the two maintained paths only:

- `LLaVA-NeXT/`: Qwen2 / LLaVA-OneVision TieDPO training and evaluation
- `Qwen3VL-TieDPO/`: Qwen3-VL TieDPO training and evaluation

Large local assets such as models, datasets, checkpoints, caches, and eval outputs are ignored by `.gitignore` and are expected to live outside git.

## Required Environment Variables

Set these before running the shell entrypoints:

```bash
export DATA_ROOT=/path/to/data_root
export MODELS_ROOT=/path/to/models_root
export HF_HOME=/path/to/hf_cache
```

Optional:

```bash
export LMMS_EVAL_DIR=/path/to/lmms-eval
export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
export WANDB_API_KEY=...
export HF_ENDPOINT=https://hf-mirror.com
```

## Qwen2

Train the maintained three-run recipe:

```bash
cd LLaVA-NeXT
bash scripts/train/run_lora_c1_minimal.sh
```

Run A/B-gap evaluation:

```bash
cd LLaVA-NeXT
bash scripts/eval/run_qwen2_7b_3model_abgap_eval.sh
```

Run lmms-eval benchmark comparison:

```bash
cd LLaVA-NeXT
LMMS_EVAL_DIR=/path/to/lmms-eval bash scripts/eval/run_3model_lmms_eval_and_summarize.sh
```

## Qwen3

Train the maintained TieDPO recipe:

```bash
cd Qwen3VL-TieDPO
bash scripts/run_qwen3vl_tiedpo_stage1.sh
```

Run A/B-gap evaluation:

```bash
cd Qwen3VL-TieDPO
bash scripts/run_qwen3vl_base_and_tiebench_eval.sh
```

Run lmms-eval benchmark comparison:

```bash
cd Qwen3VL-TieDPO
LMMS_EVAL_DIR=/path/to/lmms-eval bash scripts/run_qwen3vl_2model_mirb_eval.sh
```

## Data Format

Example sample:

```json
{
  "prompt": "What activities did the family enjoy?",
  "chosen": "The family enjoyed...",
  "rejected": "The celebration featured...",
  "is_tie": true,
  "images": ["images/train_10_0.png", "images/train_10_2.png"]
}
```

`images` are interpreted relative to `DATA_ROOT` unless the jsonl stores absolute paths.
