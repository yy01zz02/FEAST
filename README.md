# FEAST Project

This project implements the FEAST framework for optimizing image generation using various reward signals (CTR, HPS, etc.).

## Directory Structure

- **`ctr/`**: Contains the Click-Through Rate (CTR) prediction model.
  - Includes dataset handling, model architecture, training scripts, and an inference server for providing CTR rewards.

- **`fastvideo/`**: The core library for image generation and optimization.
  - `data_preprocess/`: Scripts for preprocessing data and embeddings.
  - `dataset/`: Dataset implementations for loading training data.
  - `models/`: Model definitions (e.g., Flux pipeline).
  - `rewards/`: Implementations of reward functions and reward servers (HPSv2, CTR, Qwen).
  - `utils/`: Utilities for distributed training, logging, and checkpointing.
  - `train_dpo_flux_kontext.py`: Main training script for DPO fine-tuning.

- **`scripts/`**: Shell scripts for launching various training and finetuning tasks.

- **`requirements.txt`**: List of Python dependencies.
