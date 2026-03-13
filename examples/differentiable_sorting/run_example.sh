#!/bin/bash
# Run pure gradient-based sorting algorithm discovery.
#
# This discovers sorting algorithms using only gradient descent
# (no LLM in the loop). The search:
# 1. Seeds a population of compare-and-swap networks
# 2. Optimizes parameters via gradient descent on a differentiable proxy loss
# 3. Discretizes optimized graphs to executable Python code
# 4. Evaluates correctness on random test arrays
#
# Usage:
#   bash run_example.sh           # Default: 50 iterations
#   bash run_example.sh 100       # Custom iteration count

set -e
cd "$(dirname "$0")"

ITERATIONS=${1:-50}

echo "Running pure gradient sorting discovery ($ITERATIONS iterations)..."
echo ""

uv run skydiscover-run evaluator.py \
  --config config_gradient_pure.yaml \
  --iterations "$ITERATIONS" \
  --log-level INFO \
  --output results_gradient_pure
