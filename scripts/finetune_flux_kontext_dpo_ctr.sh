#!/bin/bash
# Training script for Flux Kontext with CTR Reward
# Uses Qwen3-VL filter for image quality validation
# Configuration: 4x96GB GPUs with FSDP sharding

set -e

export WANDB_DISABLED=true
export SWANLAB_MODE=cloud

# Create necessary directories
mkdir -p images
mkdir -p data/outputs/kontext_dpo_ctr

# Install dependencies if needed
uv pip install swanlab -q


if [ ! -d "data/kontext_preprocessed" ]; then
    echo "Step 1: Preprocessing Data (Generating T5 Embeddings & VAE Latents)..."
    torchrun --nproc_per_node=4 --master_port 19004 \
        fastvideo/data_preprocess/preprocess_flux_kontext_embeddings.py \
        --jsonl_path data/kontext/train_metadata.jsonl \
        --output_dir data/kontext_preprocessed \
        --model_path data/flux-kontext \
        --height 1312 \
        --width 784
else
    echo "Step 1: Skipping preprocessing (data/kontext_preprocessed exists)"
fi

# ============================================================================
# Step 2: Start Training
# ============================================================================
echo "Step 2: Starting Flux Kontext Online DPO training with CTR..."

torchrun --nproc_per_node=4 --master_port 19009 \
    fastvideo/train_dpo_flux_kontext.py \
    --seed 42 \
    --pretrained_model_name_or_path data/flux-kontext \
    --cache_dir data/.cache \
    --data_json_path data/kontext_preprocessed/metadata.json \
    --gradient_checkpointing \
    --train_batch_size 2 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 1000 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 100 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir data/outputs/kontext_dpo_ctr \
    --final_model_dir data/outputs/kontext_dpo_ctr_final \
    --h 1312 \
    --w 784 \
    --sampling_steps 16 \
    --eta 0.3 \
    --lr_warmup_steps 10 \
    --sampler_seed 1223627 \
    --max_grad_norm 0.1 \
    --weight_decay 0.0001 \
    --shift 3.0 \
    --timestep_fraction 0.6 \
    --clip_range 1e-3 \
    --adv_clip_max 5.0 \
    --use_ema \
    --ema_decay 0.995 \
    --init_same_noise \
    --fsdp_sharding_startegy full \
    --beta 5000.0 \
    --ref_update_step 50 \
    --reward_type ctr \
    --ctr_metadata_csv data/kontext/ctr_metadata.csv \
    --qwen_filter_url http://localhost:8171 \
    --num_valid_samples 2 \
    --max_sample_attempts 20

echo "Training completed!"
echo "Model saved to: data/outputs/kontext_dpo_ctr_final"
