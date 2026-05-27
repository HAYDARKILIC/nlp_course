# Week 4 — Distributed Word Representations

> From sparse counts to dense vectors. We implement word2vec (skip-gram with negative sampling), GloVe, and fastText, each from scratch.

## Learning Objectives

- Derive the skip-gram with negative-sampling (SGNS) objective and prove its equivalence to factorizing a shifted PMI matrix (Levy & Goldberg, 2014).
- Implement SGNS in NumPy with manual gradient computation.
- Implement GloVe and explain why its objective is a weighted least-squares matrix factorization.
- Evaluate embeddings intrinsically (analogy, similarity benchmarks) and extrinsically (downstream classification).

## Required Reading

- Mikolov, T., et al. (2013). *Distributed Representations of Words and Phrases*.
- Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation*.
- Levy, O., & Goldberg, Y. (2014). *Neural Word Embedding as Implicit Matrix Factorization*.
- Bojanowski, P., et al. (2017). *Enriching Word Vectors with Subword Information*.

## Notebook

Open [`04_word_embeddings.ipynb`](04_word_embeddings.ipynb).

## Exercises

The exercise set is in the final section of the notebook. Suggested time: 4–6 hours.

## Going Further

- Browse the `assets/` folder at the repository root for figures used in lecture.
- The `utils/` package contains shared text, metrics, and visualization helpers.
