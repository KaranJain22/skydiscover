"""
Differentiable proxy loss for sorting.

Measures "sortedness" using soft pairwise comparisons.
When the output is perfectly sorted (ascending), loss is 0.

Also provides generate_inputs() for creating training data
for pure gradient-based discovery.
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


def generate_inputs(batch_size=8, n=5):
    """Generate random training inputs for gradient optimization.

    Creates a ProgramState with element-wise variables (e0, e1, ..., e{n-1})
    matching the PrimitiveGraph input format used by seed graphs.

    Args:
        batch_size: Number of examples per batch.
        n: Number of elements to sort.

    Returns:
        ProgramState with random element variables.
    """
    from skydiscover.search.differentiable.primitives.base import ProgramState

    state = ProgramState(batch_size=batch_size)
    # Set element-wise variables for PrimitiveGraph compatibility
    for i in range(n):
        state[f"e{i}"] = torch.randn(batch_size)
    # Also set combined array for proxy_loss compatibility
    state["array"] = torch.stack([state[f"e{i}"] for i in range(n)], dim=1)
    return state
