# Datasets

A curated registry of datasets used across the course. Most are loaded on
demand from Hugging Face Datasets; a handful (Shakespeare, small samples)
are vendored in subfolders for reproducibility.

| Week | Dataset | Source | Purpose |
|------|---------|--------|---------|
| 01 | Reuters-21578, Brown Corpus | NLTK | Empirical entropy of English |
| 02 | WikiText-103 | HF `wikitext` | Train BPE / WordPiece / Unigram |
| 03 | 20 Newsgroups, IMDb | sklearn / HF `imdb` | Bag-of-Words / TF-IDF classification |
| 04 | text8, Wikipedia dump (small) | mattmahoney.net | word2vec / GloVe / fastText training |
| 05 | Penn Treebank | HF `ptb_text_only` | Neural language modeling |
| 06 | Tiny Shakespeare | char-rnn release | Character-level LM |
| 07 | Multi30K | HF `bentrevett/multi30k` | English ↔ German translation |
| 08 | Multi30K | (same) | Transformer translation |
| 09 | GLUE, SQuAD | HF `glue`, HF `squad` | Probing pretrained encoders |
| 10 | Alpaca, OpenAssistant (sample) | HF | Instruction tuning, DPO |
| 11 | Natural Questions, MS MARCO (sample) | HF | RAG evaluation |
| 12 | Project-specific | — | Capstone |

## Loading conventions

```python
from datasets import load_dataset

ds = load_dataset("wikitext", "wikitext-103-raw-v1")
```

For datasets that require manual download (e.g. Multi30K mirrors come and
go), each week's README contains current instructions.

## Storage layout

```
datasets/
├── raw/        # downloaded source files (gitignored)
├── cache/      # HF datasets cache (gitignored)
└── samples/    # small vendored samples used in notebooks
```
