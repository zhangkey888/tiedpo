# TieDPO: Tie-Aware Direct Preference Optimization for Multi-Image MLLMs
[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg)](https://arxiv.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
**Not all preferences are strict.** TieDPO extends DPO with tie-aware regularization, preserving the calibrated indifference structure in base models that standard DPO systematically destroys.
---
## TL;DR
Standard DPO assumes every response pair has a winner and a loser. In multi-image reasoning, this assumption **systematically fails**: different image subsets, visual cues, and detail levels can all lead to equally correct conclusions. Forcing these **ties** into binary labels creates spurious preference boundaries that manifest as verbosity bias, position bias, and evidence-path overcommitment.
**TieDPO** preserves standard DPO on strict pairs while regularizing tie pairs toward indifference via a margin-based penalty. Across 7 benchmarks and 2 model backbones (LLaVA-OneVision-7B, Qwen3-VL-8B), TieDPO is the **only method** that jointly: (1) improves standard benchmarks over the base model, (2) preserves and improves tie recognition, and (3) achieves positive calibration gap (CalibGap).
To support this, we release **MultiTie-22k** (22k ternary-labeled preference pairs across 4 tie categories) and **TieBench-MI** (1,200 diagnostic pairs for tie-handling evaluation).
---
## Why Ties Matter
Given a set of images and a query like "Compare the architectural styles," two responses can arrive at the same correct conclusion through **different image subsets** — one drawing evidence from {I₁, I₃} (European), the other from {I₂, I₄} (East Asian). Both are valid. Standard DPO forces a binary label, penalizing one valid evidence path. **TieDPO recognizes this as a tie and regularizes toward indifference, preserving both paths.**
Ties are not monolithic — we identify four structurally distinct categories:
| Category | Definition | Failure mode if binarized |
|----------|------------|--------------------------|
| **Detail Tie** | Same conclusion, different granularity | Verbosity bias |
| **Style/Framing Tie** | Same content, different organization | Format preference bias |
| **Evidence-Path Tie** | Same conclusion via different image subsets | Evidence-path overcommitment |
| **Order-Invariance Tie** | Same content, different presentation order | Ordering bias |
**Evidence-path ties are unique to multi-image reasoning** and are the most consequential source of spurious preference boundaries.
### The Hidden DPO Pathology
Standard DPO does not merely fail to learn ties — it **actively erodes** the tie-handling structure already present in the base model. Base models score 16–25% tie-recognition accuracy on TieBench-MI; after DPO-training on data containing structured ties, tie accuracy **collapses by up to 80%**, and several standard multi-image benchmarks fall *below the untuned base*.
---
## Method
Let $x = (\mathcal{I}, q)$ be a multi-image input and $y_a, y_b$ two candidate responses. The relative preference logit is:
$$\Delta_\theta = \bigl(\log\pi_\theta(y_a|x) - \log\pi_\theta(y_b|x)\bigr) - \bigl(\log\pi_{\text{ref}}(y_a|x) - \log\pi_{\text{ref}}(y_b|x)\bigr)$$
The TieDPO objective combines three terms applied on **disjoint** data subsets:
$$\mathcal{L} = \underbrace{\alpha_{\text{dpo}} \cdot \mathbf{1}[r\neq 0] \cdot \mathcal{L}_{\text{strict}}}_{\text{discriminate strict pairs}} \;+\; \underbrace{\lambda_{\text{tie}} \cdot \mathbf{1}[r=0] \cdot \mathcal{L}_{\text{tie}}}_{\text{regularize tie pairs}} \;+\; \underbrace{\lambda_{\text{sft}} \cdot \mathcal{L}_{\text{sft}}}_{\text{SFT anchor}}$$
- **$\mathcal{L}_{\text{strict}}$**: Standard DPO sigmoid loss, identical to vanilla DPO on strict pairs — pushes chosen/rejected apart.
- **$\mathcal{L}_{\text{tie}}$**: Margin-based penalty $\max(0, |\beta \cdot \Delta_\theta| - m)^2$ — pulls tie pairs toward an indifference zone. Default $m=0$ (pure quadratic pull-to-zero).
- **$\mathcal{L}_{\text{sft}}$**: SFT anchor on chosen responses — preserves generation quality when tie regularization attenuates the discriminative signal.
**Design target**: calibration preservation, not margin maximization. TieDPO deliberately produces a moderate Strict-Margin ($0.40$–$0.53$) rather than the saturated one DPO optimizes for ($1.15$–$1.17$), because the saturated-margin regime is what collapses base-model tie structure.
---
## Key Results
### Standard Benchmarks (7 benchmarks, 2 backbones)
TieDPO is the **only method that turns preference data into consistent gains** across both LLaVA-OV-7B and Qwen3-VL-8B. The 7-benchmark average improves from 51.5→54.6 (LLaVA-OV) and 67.9→69.0 (Qwen3-VL). Standard DPO variants fail to convert the same data: **DPO-forced degrades below the base** (−4.2 on LLaVA-OV, −4.6 on Qwen3-VL), confirming that random binary supervision on tie pairs is strictly worse than discarding them.
### TieBench-MI Diagnostics
TieDPO is the only method with **positive CalibGap on both backbones** (+0.22 LLaVA-OV, +0.37 Qwen3-VL). DPO-strict drops Tie-Acc from 18.6→6.3 (−66%) on LLaVA-OV despite achieving the largest Strict-Margin (1.15) — showing the margin is over-sharpening that buys nothing on downstream tasks.
### Ablation
| Remove this component | What happens |
|----------------------|--------------|
| $\mathcal{L}_{\text{tie}}$ | Tie-Acc collapses (23.0→12.2), CalibGap goes negative |
| $\mathcal{L}_{\text{sft}}$ | Benchmarks drop (MuirBench 48.2→41.6), generation quality degrades |
| $\mathcal{L}_{\text{strict}}$ (tie-only) | Maximizes Tie-Acc (38.0) but collapses Strict-Margin — model over-commits to indifference |
---
## Repository Structure
tiedpo/
├── LLaVA-NeXT/                  # Qwen2 / LLaVA-OneVision pipeline
│   ├── trl/trainer/             #   tie_dpo_trainer.py — core TieDPO loss
│   ├── llava/train/             #   train_tie_dpo.py — entrypoint, dataset, collator
│   │                           #   llava_trainer.py — LLaVATieDPOTrainer wrapper
│   ├── llava/eval/              #   model_vqa.py, evaluate_interleave.py
│   ├── data_processing/         #   JSON/JSONL loaders
│   ├── scripts/                 #   train & eval shell entrypoints + DeepSpeed configs
│   │   ├── train/TieDPO.sh     #   Main training launch script
│   │   └── zero3.json          #   ZeRO-3 config (also zero2, zero3_offload, etc.)
│   └── data/sample.json         #   Example data format
├── Qwen3VL-TieDPO/              # Qwen3-VL pipeline (configs, scripts)
├── paper/                       # LaTeX source
└── EXPERIMENT_PLAN.md           # Full experiment plan (Chinese)
---
## Quick Start
### Environment
```bash
conda create -n tiedpo python=3.11 -y && conda activate tiedpo
cd LLaVA-NeXT && bash setup_env.sh
Required environment variables:
export DATA_ROOT=/path/to/data_root       # root for image paths
export MODELS_ROOT=/path/to/models_root   # base model checkpoints
export HF_HOME=/path/to/hf_cache          # HuggingFace cache
Optional:
export LMMS_EVAL_DIR=/path/to/lmms-eval   # for benchmark evaluation
export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
Training (LLaVA-OneVision-7B)
Edit paths in scripts/train/TieDPO.sh, then:
cd LLaVA-NeXT
bash scripts/train/TieDPO.sh
Uses LoRA (r=128), DeepSpeed ZeRO-3, 4 GPUs. Key hyperparameters:
Parameter
--beta
--dpo_alpha
--lambda_tie
--tie_margin
--gamma
--lora_r
Training (Qwen3-VL-8B)
cd Qwen3VL-TieDPO
bash scripts/run_qwen3vl_tiedpo_stage1.sh
Evaluation
A/B-gap evaluation (LLaVA-OV):
cd LLaVA-NeXT
bash scripts/eval/run_qwen2_7b_3model_abgap_eval.sh
lmms-eval benchmarks:
cd LLaVA-NeXT
LMMS_EVAL_DIR=/path/to/lmms-eval \
bash scripts/eval/run_3model_lmms_eval_and_summarize.sh
TieBench-MI evaluation (Qwen3-VL):
cd Qwen3VL-TieDPO
bash scripts/run_qwen3vl_base_and_tiebench_eval.sh
---
Data Format
Each sample in the training JSON/JSONL:
{
  "prompt": "What activities did the family enjoy?",
  "chosen": "The family enjoyed a picnic, played frisbee, and went boating.",
  "rejected": "The celebration featured outdoor games and water activities.",
  "is_tie": true,
  "images": ["images/train_10_0.png", "images/train_10_2.png"]
}
Field
prompt
chosen
rejected
is_tie
images
Ternary labels (better_a, better_b, tie) are also supported. When label=better_b, chosen/rejected are swapped internally so the model always sees chosen=better. For tie pairs, is_tie=True and ordering is arbitrary.
---
Citation
@inproceedings{tiedpo2026,
  title     = {Not All Preferences Are Strict: Tie-Aware Preference Optimization for Multi-Image MLLMs},
  author    = {Anonymous},
  booktitle = {NeurIPS},
  year      = {2026}
}
License
Apache 2.0
