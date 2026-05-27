"""From-scratch text-processing helpers used across the course."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, Iterator


def normalize(text: str, lower: bool = True, form: str = "NFC") -> str:
    """Unicode-normalize text and optionally lowercase.

    Parameters
    ----------
    text : str
        Input string.
    lower : bool, default True
        Whether to apply lowercasing. Set False for case-sensitive tasks
        (NER, POS tagging, code).
    form : str
        Unicode normalization form. Use ``NFC`` for most text; ``NFKC`` if
        you want compatibility decomposition (e.g., fullwidth → ASCII).
    """
    text = unicodedata.normalize(form, text)
    if lower:
        text = text.lower()
    return text


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def whitespace_tokenize(text: str) -> list[str]:
    """Trivial baseline tokenizer. Use only as a comparison reference."""
    return text.split()


def regex_tokenize(text: str) -> list[str]:
    """Slightly smarter tokenizer that splits punctuation from words."""
    return _TOKEN_RE.findall(text)


def ngrams(tokens: Iterable[str], n: int) -> Iterator[tuple[str, ...]]:
    """Yield consecutive n-grams from a token iterable."""
    tokens = list(tokens)
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def vocab_from_counts(counter: Counter, min_count: int = 1, max_size: int | None = None) -> dict[str, int]:
    """Build a deterministic token -> id mapping from a Counter.

    Special tokens ``<pad>``, ``<unk>``, ``<bos>``, ``<eos>`` are reserved
    ids 0–3 by convention used throughout the course.
    """
    specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
    vocab = {tok: i for i, tok in enumerate(specials)}
    items = [(t, c) for t, c in counter.most_common() if c >= min_count]
    if max_size is not None:
        items = items[: max_size - len(specials)]
    for tok, _ in items:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab
