# Week 2 — Text Processing and Tokenization

> How a string becomes a sequence of integers. We start with whitespace splitting, then progressively rebuild every modern subword algorithm: BPE, WordPiece, Unigram (SentencePiece).

## Learning Objectives

- Explain the trade-offs between character-level, word-level, and subword tokenization.
- Implement Byte-Pair Encoding (BPE) from scratch and train it on a real corpus.
- Implement the WordPiece and Unigram LM algorithms and compare their vocabularies.
- Benchmark a from-scratch tokenizer against Hugging Face `tokenizers` on speed and compression ratio.

## Required Reading

- Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units*.
- Kudo, T. (2018). *Subword Regularization*.
- Schuster, M., & Nakajima, K. (2012). *Japanese and Korean Voice Search*.

## Notebook

Open [`02_tokenization.ipynb`](02_tokenization.ipynb).

## Exercises

The exercise set is in the final section of the notebook. Suggested time: 4–6 hours.

## Going Further

- Browse the `assets/` folder at the repository root for figures used in lecture.
- The `utils/` package contains shared text, metrics, and visualization helpers.
