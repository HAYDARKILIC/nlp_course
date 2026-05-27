"""Evaluation metrics implemented from scratch.

Production code should reach for `sacrebleu`, `rouge-score`, or
`evaluate`. These are here so you can read the math.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


def perplexity(cross_entropy_nats: float) -> float:
    """Perplexity from natural-log cross-entropy.

    Perplexity = exp(H(p, q)). Reported per token.
    """
    return math.exp(cross_entropy_nats)


def accuracy(pred: Sequence, gold: Sequence) -> float:
    if len(pred) != len(gold):
        raise ValueError("pred and gold must have the same length.")
    if not pred:
        return 0.0
    return sum(int(p == g) for p, g in zip(pred, gold)) / len(pred)


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu(reference: Sequence[str], hypothesis: Sequence[str], max_n: int = 4) -> float:
    """Sentence-level BLEU with brevity penalty.

    This is the minimal Papineni et al. (2002) formulation, sufficient for
    teaching. Use `sacrebleu` for corpus-level evaluation in research code.
    """
    if not hypothesis:
        return 0.0

    log_p = 0.0
    for n in range(1, max_n + 1):
        hyp_counts = _ngram_counts(hypothesis, n)
        ref_counts = _ngram_counts(reference, n)
        clipped = sum(min(c, ref_counts[g]) for g, c in hyp_counts.items())
        total = max(sum(hyp_counts.values()), 1)
        if clipped == 0:
            return 0.0
        log_p += math.log(clipped / total) / max_n

    bp = 1.0 if len(hypothesis) >= len(reference) else math.exp(1 - len(reference) / max(len(hypothesis), 1))
    return bp * math.exp(log_p)


def rouge_l(reference: Sequence[str], hypothesis: Sequence[str], beta: float = 1.0) -> float:
    """ROUGE-L F-measure based on longest common subsequence."""
    m, n = len(reference), len(hypothesis)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if reference[i] == hypothesis[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / n
    r = lcs / m
    return ((1 + beta**2) * p * r) / (r + beta**2 * p)
