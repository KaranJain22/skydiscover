# Differentiable Sorting Example

Demonstrates gradient-based algorithm discovery using SkyDiscover's
differentiable search modes. Discovers sorting algorithms by optimizing
compositions of soft comparison and swap primitives.

## Files

- `proxy_loss.py` - Differentiable proxy loss measuring "sortedness"
- `demo_primitives.py` - Standalone demo of differentiable primitives
- `config_differentiable.yaml` - Config for hybrid mode (LLM + gradient)
- `config_gradient_pure.yaml` - Config for pure gradient mode

## Quick Start

```bash
# Run the standalone primitives demo
python demo_primitives.py

# Run discovery with hybrid mode
skydiscover-run --config config_differentiable.yaml
```

## How It Works

1. **Soft primitives**: Comparison (SoftGT) and swap (SoftSwap) operations
   are relaxed to continuous functions via sigmoid/softmax.

2. **During search**: All operations are differentiable. Gradient descent
   optimizes parameters through the soft operations.

3. **During evaluation**: Operations are discretized (hard mode). The
   resulting algorithm is valid Python with if/swap statements.

4. **Proxy loss**: Measures how sorted the output is using differentiable
   pairwise comparisons. Zero when perfectly sorted.
