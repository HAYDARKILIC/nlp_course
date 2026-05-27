# Week 10 — Fine-Tuning, PEFT, and Alignment

> How to turn a pretrained model into something useful — efficiently, and aligned with human preferences. We cover full fine-tuning, LoRA, instruction tuning, RLHF, and DPO.

## Learning Objectives

- Fine-tune BERT on a classification task and discuss catastrophic forgetting.
- Derive Low-Rank Adaptation (LoRA, Hu et al., 2021) and implement it from scratch as a PyTorch module.
- Implement supervised instruction tuning on an open dataset.
- Derive the RLHF objective and contrast it with Direct Preference Optimization (DPO, Rafailov et al., 2023).

## Required Reading

- Hu, E., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*.
- Ouyang, L., et al. (2022). *Training Language Models to Follow Instructions with Human Feedback*.
- Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*.
- Christiano, P., et al. (2017). *Deep Reinforcement Learning from Human Preferences*.

## Notebook

Open [`10_finetuning_alignment.ipynb`](10_finetuning_alignment.ipynb).

## Exercises

The exercise set is in the final section of the notebook. Suggested time: 4–6 hours.

## Going Further

- Browse the `assets/` folder at the repository root for figures used in lecture.
- The `utils/` package contains shared text, metrics, and visualization helpers.
