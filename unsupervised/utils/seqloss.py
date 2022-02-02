
import torch
import torch.nn as nn

def SequenceLoss(logits,
                  targets,
                  weights,
                  average_across_timesteps=True,
                  average_across_batch=True):
    """Weighted cross-entropy loss for a sequence of logits.

      Depending on the values of `average_across_timesteps` and
      `average_across_batch`, the return Tensor will have rank 0, 1, or 2 as these
      arguments reduce the cross-entropy at each target, which has shape
      `[batch_size, sequence_length]`, over their respective dimensions. For
      example, if `average_across_timesteps` is `True` and `average_across_batch`
      is `False`, then the return Tensor will have shape `[batch_size]`.

      Args:
        logits: A Tensor of shape
          `[batch_size, sequence_length, num_decoder_symbols]` and dtype float.
          The logits correspond to the prediction across all classes at each
          timestep.
        targets: A Tensor of shape `[batch_size, sequence_length]` and dtype
          int. The target represents the true class at each timestep.
        weights: A Tensor of shape `[batch_size, sequence_length]` and dtype
          float. `weights` constitutes the weighting of each prediction in the
          sequence. When using `weights` as masking, set all valid timesteps to 1
          and all padded timesteps to 0, e.g. a mask returned by `tf.sequence_mask`.
        average_across_timesteps: If set, sum the cost across the sequence
          dimension and divide the cost by the total label weight across timesteps.
        average_across_batch: If set, sum the cost across the batch dimension and
          divide the returned cost by the batch size.
        softmax_loss_function: Function (labels, logits) -> loss-batch
          to be used instead of the standard softmax (the default if this is None).
          **Note that to avoid confusion, it is required for the function to accept
          named arguments.**
        name: Optional name for this operation, defaults to "sequence_loss".

      Returns:
        A float Tensor of rank 0, 1, or 2 depending on the
        `average_across_timesteps` and `average_across_batch` arguments. By default,
        it has rank 0 (scalar) and is the weighted average cross-entropy
        (log-perplexity) per symbol.

      Raises:
        ValueError: logits does not have 3 dimensions or targets does not have 2
                    dimensions or weights does not have 2 dimensions.
      """
    if len(logits.size()) != 3:
        raise ValueError("Logits must be a "
                         "[batch_size x sequence_length x logits] tensor")
    if len(targets.size()) != 2:
        raise ValueError("Targets must be a [batch_size x sequence_length] "
                         "tensor")
    if len(weights.size()) != 2:
        raise ValueError("Weights must be a [batch_size x sequence_length] "
                         "tensor")
    num_classes = logits.size(2)
    logits_flat = logits.view(-1, num_classes)
    targets = targets.view(-1)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    crossent = loss_fn(logits_flat, targets)
    if average_across_timesteps and average_across_batch:
        crossent = crossent.sum()
        total_size = weights.sum()
        total_size += 1e-12  # to avoid division by 0 for all-0 weights
        crossent /= total_size
    else:
        batch_size = logits.size(0)
        sequence_length = logits.size(1)
        crossent = crossent.view(batch_size, sequence_length)
    if average_across_timesteps and (not average_across_batch):
        crossent = crossent.sum(dim=1)
        total_size = weights.sum(dim=1)
        total_size += 1e-12  # to avoid division by 0 for all-0 weights
        crossent /= total_size
    if (not average_across_timesteps) and average_across_batch:
        crossent = crossent.sum(dim=0)
        total_size = weights.sum(dim=0)
        total_size += 1e-12  # to avoid division by 0 for all-0 weights
        crossent /= total_size
    return crossent