# Natural Language Processing: From First Principles to Modern LLMs

A 12-week, research-grade course that rebuilds the entire NLP stack from scratch — starting from character-level text manipulation and ending at instruction-tuned large language models. Every concept is derived mathematically, implemented in pure **NumPy** or **PyTorch** (no high-level NLP libraries until the relevant primitive is fully understood), and stress-tested on real datasets. It is the bridge between classical statistical NLP and the modern transformer-based ecosystem.

---

## Course Philosophy

1. **Derive before you import.** Every algorithm — from TF-IDF to multi-head attention — is implemented from scratch before its production-grade counterpart (scikit-learn, Hugging Face, PyTorch built-ins) is allowed.
2. **Mathematics is non-negotiable.** Each notebook opens with the underlying linear algebra, probability, or information theory, then translates it into code.
3. **Benchmarks over intuition.** Every implementation is profiled against an industry-standard reference (gensim, fastText, Hugging Face Transformers) on both correctness and speed.
4. **Read the paper.** Each week is anchored to one or two canonical papers. The notebooks reconstruct the paper's central results.

---

## Curriculum at a Glance

| Week | Topic | Core Notebook(s) | Anchor Papers |
|------|-------|------------------|---------------|
| 01 | Foundations: linguistics, probability, information theory | `01_foundations.ipynb` | Shannon (1948), Manning & Schütze Ch. 2 |
| 02 | Text processing: tokenization, normalization, BPE, WordPiece | `02_tokenization.ipynb` | Sennrich et al. (2016), Kudo (2018) |
| 03 | Classical representations: BoW, TF-IDF, n-grams, LSA | `03_classical_representations.ipynb` | Salton (1975), Deerwester (1990) |
| 04 | Word embeddings: word2vec, GloVe, fastText, evaluation | `04_word_embeddings.ipynb` | Mikolov et al. (2013), Pennington et al. (2014) |
| 05 | Language models: n-gram → neural → perplexity | `05_language_models.ipynb` | Bengio et al. (2003) |
| 06 | RNNs, LSTMs, GRUs from scratch with BPTT | `06_rnn_lstm.ipynb` | Hochreiter & Schmidhuber (1997) |
| 07 | Seq2Seq and attention mechanisms | `07_seq2seq_attention.ipynb` | Sutskever et al. (2014), Bahdanau et al. (2015) |
| 08 | The Transformer, implemented end-to-end | `08_transformer.ipynb` | Vaswani et al. (2017) |
| 09 | Pretrained models: BERT, GPT, T5 — architectures and objectives | `09_pretrained_models.ipynb` | Devlin et al. (2019), Radford et al. (2019) |
| 10 | Fine-tuning, PEFT, LoRA, RLHF, DPO | `10_finetuning_alignment.ipynb` | Hu et al. (2021), Ouyang et al. (2022), Rafailov et al. (2023) |
| 11 | Modern LLM applications: RAG, tool use, evaluation harnesses | `11_modern_llm_applications.ipynb` | Lewis et al. (2020) |
| 12 | Capstone: build, fine-tune, evaluate, and deploy a domain-specific LM | `12_capstone.ipynb` | — |

---

## Prerequisites

- **Python** (intermediate): NumPy, basic PyTorch, virtual environments.
- **Mathematics**: linear algebra, multivariable calculus, probability (see the `probability` and `statistics` repositories in this portfolio).
- **Deep learning** (recommended): the `deep_learning` repository covers the necessary background.

---

## Repository Layout

```
nlp_course/
├── week01_foundations/
│   ├── 01_foundations.ipynb
│   └── README.md
├── week02_text_processing/
│   ├── 02_tokenization.ipynb
│   └── README.md
├── ... (weeks 03–12)
├── datasets/
│   └── README.md          # Curated dataset registry
├── utils/
│   ├── __init__.py
│   ├── text.py            # From-scratch text helpers
│   ├── metrics.py         # BLEU, ROUGE, perplexity, etc.
│   └── viz.py             # Attention maps, embedding projections
├── assets/                # Figures, diagrams
├── requirements.txt
├── environment.yml
└── LICENSE
```

---

## Environment

```bash
# With conda
conda env create -f environment.yml
conda activate nlp-course

# Or with pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Recommended hardware: a single consumer GPU (≥ 8 GB VRAM) is sufficient for weeks 1–9. Weeks 10–12 benefit from ≥ 24 GB VRAM or a cloud instance.

---

## How to Use This Course

Each week's folder contains:

- **The main notebook** — the lecture, derivations, and code.
- **A `README.md`** with learning objectives, reading list, and the exercise set.
- **Solutions** (in a hidden cell or separate file) for the exercises.

Work through the notebook cell by cell. Do not skip the from-scratch implementation — the production library calls come *after* you've built the primitive yourself.

---

## Citation

If this material is useful in your teaching or research:

```bibtex
@misc{kilic2026nlp,
  author = {Kılıç, Haydar},
  title  = {Natural Language Processing: From First Principles to Modern LLMs},
  year   = {2026},
  url    = {https://github.com/HAYDARKILIC/nlp_course}
}
```

---

## License

MIT. See `LICENSE`.
