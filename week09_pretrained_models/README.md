# Week 9 — Pretrained Models: BERT, GPT, T5

> Three architectures, three objectives, three philosophies. We dissect each, then load and probe the pretrained weights with Hugging Face.

## Learning Objectives

- Explain encoder-only, decoder-only, and encoder–decoder architectures and which tasks each suits.
- Implement the masked language modeling (MLM) and next-sentence prediction objectives.
- Implement causal language modeling and explain why it scales to arbitrary generation tasks.
- Probe BERT's internal representations layer-by-layer (Tenney et al., 2019; Rogers et al., 2020).

## Required Reading

- Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
- Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners*.
- Raffel, C., et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*.
- Rogers, A., Kovaleva, O., & Rumshisky, A. (2020). *A Primer in BERTology*.

## Notebook

Open [`09_pretrained_models.ipynb`](09_pretrained_models.ipynb).

## Exercises

The exercise set is in the final section of the notebook. Suggested time: 4–6 hours.

## Going Further

- Browse the `assets/` folder at the repository root for figures used in lecture.
- The `utils/` package contains shared text, metrics, and visualization helpers.
