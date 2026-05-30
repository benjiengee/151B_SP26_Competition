# GRPO Fine-Tuning Plan

Goal: implement a new Jupyter notebook, `grpo_train.ipynb`, on the `benji-grpo` branch to fine-tune `Qwen3-4B-Thinking-2507` for math reasoning using TRL GRPO.

## Checklist

- [x] Check current git status so unrelated work is not disturbed.
- [x] Switch to the existing `benji-grpo` branch, or create it if it does not exist.
- [x] Confirm the notebook will be created as `grpo_train.ipynb`.
- [x] Inspect the project structure for existing notebooks, training scripts, configs, dataset paths, and dependency files.
- [x] Identify whether the repo already uses Hugging Face, PyTorch, TRL, PEFT/LoRA, Accelerate, or Weights & Biases.
- [x] Add notebook sections for environment checks, imports, config, model/tokenizer loading, dataset loading, reward functions, GRPO setup, training, evaluation, and saving.
- [x] Configure `Qwen3-4B-Thinking-2507` loading through Hugging Face Transformers.
- [x] Decide appropriate dtype, device mapping, and memory strategy.
- [x] Add optional LoRA/QLoRA support if the repo or available GPU constraints suggest it is needed.
- [x] Determine the math reasoning dataset source.
- [x] Format examples into prompts suitable for Qwen thinking/reasoning behavior.
- [x] Include train/eval split handling.
- [x] Add a small debug subset option for quick smoke tests.
- [x] Implement reward functions compatible with TRL GRPO.
- [x] Add answer correctness reward.
- [x] Add math formatting or final-answer extraction reward.
- [x] Add optional reasoning-structure reward if useful.
- [x] Add helper utilities for extracting boxed or final answers.
- [x] Use `trl.GRPOTrainer` and `GRPOConfig`.
- [x] Configure generation count, max prompt/completion lengths, batch sizes, gradient accumulation, learning rate, logging, checkpointing, and eval cadence.
- [x] Keep training settings easy to edit near the top of the notebook.
- [x] Add a tiny dry-run path to verify dataset loading, model/tokenizer loading, reward execution, and trainer initialization.
- [x] Avoid launching a full training run automatically.
- [x] Update or document required packages such as `trl`, `transformers`, `accelerate`, `datasets`, `peft`, `bitsandbytes`, and `math_verify` if needed.
- [x] Prefer notebook-local install/check cells unless the repo already has dependency management.
- [x] Run notebook JSON validation.
- [x] Verify the notebook opens structurally.
- [x] Summarize created files and assumptions, especially around dataset choice and compute requirements.
