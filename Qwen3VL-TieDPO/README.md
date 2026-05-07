# Qwen3VL-TieDPO

Maintained files in this directory cover only:

- Qwen3-VL TieDPO training
- Qwen3-VL A/B-gap evaluation
- Qwen3-VL lmms-eval benchmark comparison

## Required Environment Variables

```bash
export DATA_ROOT=/path/to/data_root
export HF_HOME=/path/to/hf_cache
```

Optional:

```bash
export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
export LMMS_EVAL_DIR=/path/to/lmms-eval
export HF_ENDPOINT=https://hf-mirror.com
```

## Training

```bash
cd Qwen3VL-TieDPO
bash scripts/run_qwen3vl_tiedpo_stage1.sh
```

Important defaults:

- base model: `models/Qwen3-VL-8B-Instruct`
- output dir: `workspace/outputs/`
- train / eval jsonl: `${DATA_ROOT}/processed_splits_balanced_16k_with_evidence/...`

## A/B-gap Evaluation

Run base only:

```bash
cd Qwen3VL-TieDPO
bash scripts/run_qwen3vl_base_tiebench_eval.sh
```

Run base vs LoRA:

```bash
cd Qwen3VL-TieDPO
TIEDPO_CKPT=/path/to/lora_ckpt \
bash scripts/run_qwen3vl_base_and_tiebench_eval.sh
```

## Benchmark Evaluation

Run the maintained lmms-eval comparison:

```bash
cd Qwen3VL-TieDPO
LMMS_EVAL_DIR=/path/to/lmms-eval \
TIEDPO_CKPT=/path/to/lora_ckpt \
bash scripts/run_qwen3vl_2model_mirb_eval.sh
```

## Core Code

- `qwen3vl_tiedpo/data.py`
- `qwen3vl_tiedpo/loss.py`
- `qwen3vl_tiedpo/run_tie_dpo.py`
- `qwen3vl_tiedpo/trainer.py`
