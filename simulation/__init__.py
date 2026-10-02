"""Monte-Carlo sampling of syndromes from Stim circuits."""

from .sampler import (
    sample_syndromes,
    sample_syndromes_and_errors,
    print_sample_examples,
)

__all__ = [
    "sample_syndromes",
    "sample_syndromes_and_errors",
    "print_sample_examples",
]
