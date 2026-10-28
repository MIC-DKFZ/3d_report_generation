from typing import Callable, Optional, Union

import torch


def compute_logits_and_loss(
    hidden_states: torch.Tensor,
    labels: Optional[torch.LongTensor],
    logits_to_keep: Union[int, torch.Tensor],
    lm_head: Callable[[torch.Tensor], torch.Tensor],
    loss_function: Callable,
    vocab_size: int,
    num_items_in_batch: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply the LM head and compute the causal LM loss.

    With labels, the LM head is applied only where a target exists (the response), skipping the vision
    and prompt positions. Logits are then (num_targets, vocab) instead of (batch, sequence, vocab).
    Without labels (generation), logits follow the upstream 'logits_to_keep' slicing.
    """
    if labels is None:
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        return lm_head(hidden_states[:, slice_indices, :]), None

    shift_labels = labels[:, 1:]
    has_target = shift_labels != -100
    logits = lm_head(hidden_states[:, :-1][has_target])
    loss = loss_function(
        logits=logits,
        labels=None,
        shift_labels=shift_labels[has_target],
        vocab_size=vocab_size,
        num_items_in_batch=num_items_in_batch,
    )
    return logits, loss
