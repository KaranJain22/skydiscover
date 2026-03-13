#!/usr/bin/env python3
"""
Standalone demo of differentiable primitives for sorting.

Shows how soft primitives work in both soft (differentiable) and
hard (discrete) modes, and demonstrates gradient flow through
a simple sorting network.
"""

import sys
import os

# Add the skydiscover package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from skydiscover.search.differentiable.primitives.base import ProgramState
from skydiscover.search.differentiable.primitives.conditions import SoftGT
from skydiscover.search.differentiable.primitives.functions import SoftSwap
from skydiscover.search.differentiable.primitives.control_flow import SoftFor
from skydiscover.search.differentiable.primitives.composition import (
    PrimitiveGraph,
    Sequence,
)
from skydiscover.search.differentiable.discretizer import discretize_graph


def demo_soft_conditions():
    """Show how soft conditions produce probabilities vs hard decisions."""
    print("=" * 60)
    print("Demo 1: Soft vs Hard Conditions")
    print("=" * 60)

    state = ProgramState(batch_size=1)
    state["a"] = torch.tensor([3.0])
    state["b"] = torch.tensor([1.0])

    for beta in [1.0, 5.0, 10.0, 50.0]:
        gt = SoftGT("a", "b", beta=beta)
        p = gt(state)
        print(f"  a=3, b=1, beta={beta:5.1f} -> P(a > b) = {p.item():.6f}")

    # Hard mode
    gt_hard = SoftGT("a", "b", beta=10.0, hard=True)
    p_hard = gt_hard(state)
    print(f"  Hard mode: P(a > b) = {p_hard.item():.1f}")
    print()


def demo_soft_swap():
    """Show differentiable conditional swap."""
    print("=" * 60)
    print("Demo 2: Differentiable Conditional Swap")
    print("=" * 60)

    state = ProgramState(batch_size=1)
    state["x"] = torch.tensor([5.0], requires_grad=True)
    state["y"] = torch.tensor([2.0], requires_grad=True)

    condition = SoftGT("x", "y", beta=10.0)
    swap = SoftSwap("x", "y", condition, beta=10.0)

    print(f"  Before swap: x={state['x'].item():.1f}, y={state['y'].item():.1f}")
    state = swap(state)
    print(f"  After soft swap: x={state['x'].item():.4f}, y={state['y'].item():.4f}")
    print(f"  (x and y are approximately swapped because x > y)")
    print()


def demo_gradient_flow():
    """Show that gradients flow through soft primitives."""
    print("=" * 60)
    print("Demo 3: Gradient Flow Through Soft Primitives")
    print("=" * 60)

    # Create a learnable parameter
    threshold = torch.nn.Parameter(torch.tensor([3.0]))

    state = ProgramState(batch_size=4)
    state["values"] = torch.tensor([1.0, 2.0, 4.0, 5.0])
    state["threshold"] = threshold.expand(4)

    # Soft condition: values > threshold
    gt = SoftGT("values", "threshold", beta=5.0)
    probs = gt(state)

    # Loss: we want all probabilities to be 1 (all values > threshold)
    loss = (1 - probs).mean()

    loss.backward()
    print(f"  Threshold = {threshold.item():.2f}")
    print(f"  P(values > threshold) = {probs.detach().numpy()}")
    print(f"  Loss = {loss.item():.4f}")
    print(f"  d(loss)/d(threshold) = {threshold.grad.item():.4f}")
    print(f"  (Negative gradient -> threshold should decrease to make all values pass)")
    print()


def demo_codegen():
    """Show code generation from a primitive graph."""
    print("=" * 60)
    print("Demo 4: Code Generation from Primitive Graph")
    print("=" * 60)

    # Build a simple sorting network using SoftSwap
    condition = SoftGT("a", "b", beta=10.0)
    swap = SoftSwap("a", "b", condition, beta=10.0)

    graph = PrimitiveGraph(
        inputs=["a", "b"],
        outputs=["a", "b"],
        operations=[swap],
    )

    # Generate discrete code
    code = discretize_graph(graph)
    print("  Generated code:")
    for line in code.split("\n"):
        print(f"    {line}")
    print()


def demo_sorting_optimization():
    """Optimize a simple 3-element sorting network via gradient descent."""
    print("=" * 60)
    print("Demo 5: Gradient Optimization of Sorting Network")
    print("=" * 60)

    # Build a compare-and-swap network for 3 elements
    # Optimal: compare (0,1), (1,2), (0,1)
    cond_01 = SoftGT("e0", "e1", beta=10.0)
    swap_01 = SoftSwap("e0", "e1", cond_01, beta=10.0)

    cond_12 = SoftGT("e1", "e2", beta=10.0)
    swap_12 = SoftSwap("e1", "e2", cond_12, beta=10.0)

    cond_01b = SoftGT("e0", "e1", beta=10.0)
    swap_01b = SoftSwap("e0", "e1", cond_01b, beta=10.0)

    graph = PrimitiveGraph(
        inputs=["e0", "e1", "e2"],
        outputs=["e0", "e1", "e2"],
        operations=[swap_01, swap_12, swap_01b],
    )

    # Test: sort [3, 1, 2]
    state = ProgramState(batch_size=1)
    state["e0"] = torch.tensor([3.0])
    state["e1"] = torch.tensor([1.0])
    state["e2"] = torch.tensor([2.0])

    print(f"  Input: [{state['e0'].item():.0f}, {state['e1'].item():.0f}, {state['e2'].item():.0f}]")

    # Soft pass
    result = graph(state)
    print(f"  Soft output: [{result['e0'].item():.3f}, {result['e1'].item():.3f}, {result['e2'].item():.3f}]")

    # Hard pass
    graph.set_hard_recursive(True)
    state2 = ProgramState(batch_size=1)
    state2["e0"] = torch.tensor([3.0])
    state2["e1"] = torch.tensor([1.0])
    state2["e2"] = torch.tensor([2.0])
    result2 = graph(state2)
    print(f"  Hard output: [{result2['e0'].item():.0f}, {result2['e1'].item():.0f}, {result2['e2'].item():.0f}]")

    # Generate code
    code = discretize_graph(graph)
    print(f"\n  Generated sorting code:")
    for line in code.split("\n"):
        print(f"    {line}")
    print()


if __name__ == "__main__":
    demo_soft_conditions()
    demo_soft_swap()
    demo_gradient_flow()
    demo_codegen()
    demo_sorting_optimization()
    print("All demos completed successfully!")
