# Week 5 — Language Modeling: From n-grams to Neural

> We formalize the language modeling task, define perplexity, and trace the architectural evolution from count-based n-grams to feed-forward neural language models (Bengio et al., 2003).

## Learning Objectives

- Define a language model formally and explain its relationship to the chain rule of probability.
- Derive perplexity from cross-entropy and explain its scale (e.g., perplexity = 50 means 'the model is as confused as if choosing uniformly from 50 words').
- Implement a Kneser–Ney–smoothed n-gram LM and benchmark on a held-out corpus.
- Implement the Bengio (2003) neural language model end-to-end in PyTorch.

## Required Reading

- Bengio, Y., et al. (2003). *A Neural Probabilistic Language Model*.
- Jurafsky & Martin, Chapters 3 and 7.

## Notebook

Open [`05_language_models.ipynb`](05_language_models.ipynb).

## Exercises

The exercise set is in the final section of the notebook. Suggested time: 4–6 hours.

## Going Further

- Browse the `assets/` folder at the repository root for figures used in lecture.
- The `utils/` package contains shared text, metrics, and visualization helpers.
