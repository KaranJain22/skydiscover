"""
Differentiable proxy loss for sorting.

Measures "sortedness" using soft pairwise comparisons.
When the output is perfectly sorted (ascending), loss is 0.
"""

import torch


def proxy_loss(state):
    """Differentiable measure of how unsorted an array is.

    Computes soft inversions: for each pair (i, j) where i < j,
    adds sigmoid(beta * (array[i] - array[j])) which is ~1 when
    array[i] > array[j] (an inversion) and ~0 otherwise.

    Args:
        state: ProgramState with 'array' key containing the output.

    Returns:
        Scalar loss tensor (lower = more sorted).
    """
    array = state["array"]  # shape: (batch_size, n) or (n,)

    if array.dim() == 1:
        array = array.unsqueeze(0)

    batch_size, n = array.shape
    beta = 10.0

    # Pairwise inversion loss
    loss = torch.tensor(0.0, device=array.device)
    for i in range(n - 1):
        for j in range(i + 1, n):
            # Soft inversion: high when array[i] > array[j]
            diff = array[:, i] - array[:, j]
            loss = loss + torch.sigmoid(beta * diff).mean()

    # Normalize by number of pairs
    num_pairs = n * (n - 1) / 2
    return loss / max(num_pairs, 1)
