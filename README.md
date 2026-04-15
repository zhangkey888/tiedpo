# TieDPO: Tie-Aware Direct Preference Optimization for Multimodal LLMs

TieDPO extends DPO to handle **tie** annotations in preference data, where two responses are of equal quality. Standard DPO treats all pairs as strict preferences, which can mislead training when the two responses are actually comparable. TieDPO introduces a tie regularization loss that penalizes the model for assigning large reward margins to tied pairs.

Built on top of [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) with [LLaVA-OneVision-Qwen2-7B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov) as the base model.

## Method

The training objective combines three terms:

```
L = alpha * L_DPO + lambda_tie * L_tie + L_SFT
```

- **L_DPO**: standard DPO loss on strict (non-tie) pairs
- **L_tie**: tie regularization — penalizes large reward margins on tied pairs
- **L_SFT**: SFT loss on the chosen response

For tied pairs, the tie loss is:
```
L_tie = clamp(|reward_margin| - tie_margin, min=0)^2
```

## Data Format

Training data is a JSON array. Each sample must contain:

```json
{
  "prompt": "What activities did the family enjoy?",
  "chosen": "The family enjoyed...",
  "rejected": "The celebration featured...",
  "is_tie": true,
  "images": ["images/train_10_0.png", "images/train_10_2.png"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | string | User question |
| `chosen` | string | Preferred (or tied) response |
| `rejected` | string | Rejected (or tied) response |
| `is_tie` | bool | `true` if the two responses are of equal quality |
| `images` | list[str] | Paths relative to `--image_folder` |

A sample data file is provided at `data/sample.json`.

## Environment Setup

Requires Python 3.11 and CUDA. Tested on 8x A100/H100 GPUs.

```bash
conda create -n tiedpo python=3.11 -y
conda activate tiedpo
bash setup_env.sh
```

`setup_env.sh` installs all dependencies from `requirements.txt` and patches the local `trl/trainer/` files into the installed trl package.

## Training

Edit `scripts/train/TieDPO.sh` to set your paths:

```bash
SFT_MODEL="/path/to/llava-onevision-qwen2-7b-ov"   # base model
DATA_PATH="/path/to/tie_dpo_train.json"              # training data
IMAGE_FOLDER="/path/to/images/"                      # image root directory
```

Then run:

```bash
cd LLaVA-NeXT
bash scripts/train/TieDPO.sh
```

Key hyperparameters:

| Argument | Default | Description |
|----------|---------|-------------|
| `--beta` | 0.1 | DPO temperature |
| `--lambda_tie` | 1.0 | Weight of tie regularization loss |
| `--tie_margin` | 0.0 | Margin threshold for tie loss |
| `--dpo_alpha` | 1.0 | Weight of DPO loss |
| `--lora_r` | 128 | LoRA rank |
| `--lora_alpha` | 256 | LoRA alpha |

## Key Changes vs LLaVA-NeXT

- `trl/trainer/tie_dpo_trainer.py`: TieDPOTrainer with tie-aware loss
- `llava/train/train_tie_dpo.py`: training entry point
- `llava/train/llava_trainer.py`: LLaVATieDPOTrainer wrapping TieDPOTrainer

## Requirements

See `requirements.txt`. Core dependencies:

- PyTorch 2.8.0
- Transformers 4.57.6
- PEFT 0.18.1
- DeepSpeed 0.18.8
- Accelerate 1.12.0
