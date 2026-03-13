"""
Gradient-enhanced evolution for SkyDiscover.

Standard LLM-based evolution with gradient-based local refinement.
After LLM generates a candidate, continuous parameters are fine-tuned
via gradient descent before evaluation.
"""
