"""Training-free visual defect inspection for stamped metal brackets.

Frozen DINOv2 patch features + a PCA subspace of normal variation. No gradient
training anywhere: a per-SKU "model" is a mean vector and an eigenbasis.
"""

__version__ = "0.1.0"
