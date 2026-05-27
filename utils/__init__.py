"""Shared utilities for the NLP course.

Modules
-------
text     -- From-scratch text-processing helpers (normalization, n-grams, ...).
metrics  -- Evaluation metrics: perplexity, BLEU, ROUGE, accuracy.
viz      -- Visualization helpers for embeddings, attention maps, training curves.
"""
from . import text, metrics, viz  # noqa: F401

__all__ = ["text", "metrics", "viz"]
