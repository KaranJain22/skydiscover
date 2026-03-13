"""
Pure gradient search algorithm for SkyDiscover.

Fully differentiable search over a primitive library with no LLM in the
inner loop. Uses gradient descent on both architecture weights (WeightedChoice)
and continuous parameters.
"""
