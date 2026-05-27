# Week 8 — The Transformer, Built End-to-End

> The architecture that changed everything. We implement the full Transformer from Vaswani et al. (2017) in PyTorch — multi-head attention, positional encoding, residuals, layer norm, the works — without using `nn.Transformer`.

## Learning Objectives

- Derive scaled dot-product attention and explain the $\sqrt{d_k}$ scaling.
- Implement multi-head attention from scratch in PyTorch.
- Implement sinusoidal positional encoding and explain its rotational structure; introduce RoPE and ALiBi as modern alternatives.
- Assemble and train a full encoder–decoder Transformer on a translation task.

## Required Reading

- Vaswani, A., et al. (2017). *Attention Is All You Need*.
- Su, J., et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*.
- Press, O., Smith, N. A., & Lewis, M. (2022). *Train Short, Test Long: Attention with Linear Biases*.

## Notebook

Open [`08_transformer.ipynb`](08_transformer.ipynb).

## Exercises

The exercise set is in the final section of the notebook. Suggested time: 4–6 hours.

## Going Further

- Browse the `assets/` folder at the repository root for figures used in lecture.
- The `utils/` package contains shared text, metrics, and visualization helpers.
