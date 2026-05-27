# Week 6 — Recurrent Architectures: RNN, LSTM, GRU

> Sequential models from first principles. We derive backpropagation through time, diagnose the vanishing-gradient problem analytically, and build LSTM and GRU cells from scratch.

## Learning Objectives

- Derive backpropagation through time (BPTT) by hand for a simple RNN.
- Prove the vanishing-gradient theorem (Bengio et al., 1994).
- Implement an LSTM cell from scratch in NumPy and verify gradients against PyTorch autograd.
- Train a character-level RNN on Shakespeare (Karpathy's classic setup) and generate samples.

## Required Reading

- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*.
- Bengio, Y., Simard, P., & Frasconi, P. (1994). *Learning Long-Term Dependencies with Gradient Descent Is Difficult*.
- Cho, K., et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder*.

## Notebook

Open [`06_rnn_lstm.ipynb`](06_rnn_lstm.ipynb).

## Exercises

The exercise set is in the final section of the notebook. Suggested time: 4–6 hours.

## Going Further

- Browse the `assets/` folder at the repository root for figures used in lecture.
- The `utils/` package contains shared text, metrics, and visualization helpers.
