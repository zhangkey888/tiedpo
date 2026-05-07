from typing import Dict

import torch
import torch.nn.functional as F


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    label_pad_token_id: int = -100,
) -> torch.FloatTensor:
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits and labels must have the same batch/sequence shape.")

    shifted_labels = labels[:, 1:].clone()
    shifted_logits = logits[:, :-1, :]
    loss_mask = shifted_labels != label_pad_token_id
    shifted_labels[shifted_labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(
        shifted_logits.log_softmax(-1), dim=2, index=shifted_labels.unsqueeze(2)
    ).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1)


def get_sft_loss(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    label_pad_token_id: int = -100,
) -> torch.FloatTensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=label_pad_token_id,
    )


def get_sft_loss_per_sample(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    label_pad_token_id: int = -100,
) -> torch.FloatTensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = shift_logits.size(-1)
    token_losses = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=label_pad_token_id,
        reduction="none",
    ).view(shift_labels.size(0), shift_labels.size(1))
    valid_mask = (shift_labels != label_pad_token_id).to(token_losses.dtype)
    token_loss_sum = (token_losses * valid_mask).sum(dim=1)
    token_count = valid_mask.sum(dim=1).clamp_min(1.0)
    return token_loss_sum / token_count


def compute_tie_dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    is_tie: torch.BoolTensor,
    beta: float,
    label_smoothing: float = 0.0,
    tie_margin: float = 0.0,
    reference_free: bool = False,
) -> Dict[str, torch.Tensor]:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = (
        torch.zeros_like(pi_logratios)
        if reference_free
        else reference_chosen_logps - reference_rejected_logps
    )
    delta = pi_logratios - ref_logratios
    scaled_delta = beta * delta

    strict_mask = ~is_tie
    tie_mask = is_tie
    losses = torch.zeros_like(delta)

    if strict_mask.any():
        strict_losses = (
            -F.logsigmoid(scaled_delta[strict_mask]) * (1 - label_smoothing)
            - F.logsigmoid(-scaled_delta[strict_mask]) * label_smoothing
        )
        losses[strict_mask] = strict_losses

    if tie_mask.any():
        abs_scaled = scaled_delta[tie_mask].abs()
        if tie_margin > 0:
            tie_losses = torch.clamp(abs_scaled - tie_margin, min=0.0) ** 2
        else:
            tie_losses = abs_scaled**2
        losses[tie_mask] = tie_losses

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    device = policy_chosen_logps.device
    strict_loss = (
        losses[strict_mask].sum()
        if strict_mask.any()
        else torch.tensor(0.0, device=device)
    )
    tie_loss = losses[tie_mask].sum() if tie_mask.any() else torch.tensor(0.0, device=device)

    return {
        "strict_loss": strict_loss,
        "tie_loss": tie_loss,
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
        "margins": chosen_rewards - rejected_rewards,
    }
