# TieDPO

This repository is a cleaned, GitHub-friendly snapshot of the maintained TieDPO codepaths only.

- `LLaVA-NeXT/`: Qwen2 / LLaVA-OneVision TieDPO training, evaluation, and lmms-eval comparison
- `Qwen3VL-TieDPO/`: Qwen3-VL TieDPO training, evaluation, and lmms-eval comparison

Large local assets such as models, datasets, checkpoints, caches, merged weights, and evaluation outputs are intentionally ignored by `.gitignore`.

## Repository Scope

The curated repo keeps the parts that are useful for reproduction and inspection:

- training entrypoints and trainer implementations
- dataset loading / lightweight data construction helpers
- evaluation scripts for A/B-gap and lmms-eval
- minimal configs needed to run training or evaluation

It intentionally excludes local data, model weights, caches, and one-off environment bootstrap scripts.

## Layout

```text
tiedpo/
├── LLaVA-NeXT/
│   ├── data_processing/          # json/jsonl helpers used by training
│   ├── llava/train/              # Qwen2 / LLaVA-OneVision TieDPO training code
│   ├── trl/trainer/              # custom TieDPO trainer variants
│   └── scripts/                  # maintained train/eval entrypoints
├── Qwen3VL-TieDPO/
│   ├── qwen3vl_tiedpo/           # dataset, loss, trainer, run entrypoint
│   ├── configs/                  # FSDP / DeepSpeed configs
│   └── scripts/                  # maintained train/eval entrypoints
└── .gitignore                    # ignores local assets and generated results
```

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
```

Private API keys, access tokens, or custom model endpoints should be set only in your local shell or CI secrets, not committed to the repository.

## Maintained Entrypoints

### Qwen2 / LLaVA-OneVision

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
LMMS_EVAL_DIR=/path/to/lmms-eval \
bash scripts/eval/run_3model_lmms_eval_and_summarize.sh
```

### Qwen3-VL

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
LMMS_EVAL_DIR=/path/to/lmms-eval \
bash scripts/run_qwen3vl_2model_mirb_eval.sh
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
