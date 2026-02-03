# Copyright (c) [2025] [FastVideo Team]
# SPDX-License-Identifier: [Apache License 2.0]
#
# Training script for Flux Kontext instruction editing model with Online DPO.
# Adapted from FlowGRPO's online DPO training approach.
# Supports full fine-tuning with FSDP sharding for multi-GPU setup.
# Uses SwanLab for logging and pre-computed T5 embeddings.
# Supports Qwen3-VL based image quality filtering for valid sample selection.

import argparse
import math
import os
from pathlib import Path
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.communications_flux import sp_parallel_dataloader_wrapper
import time
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions

from torch.utils.data.distributed import DistributedSampler
import swanlab
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.training_utils import compute_density_for_timestep_sampling
from fastvideo.dataset.latent_flux_kontext_rl_datasets import (
    FluxKontextLatentDataset,
    flux_kontext_latent_collate_function,
)
from fastvideo.rewards import get_reward_fn, QwenImageFilter
import torch.distributed as dist
from fastvideo.utils.checkpoint import save_checkpoint, save_final_model
from fastvideo.utils.logging_ import main_print
from diffusers.image_processor import VaeImageProcessor

check_min_version("0.31.0")
from collections import deque, defaultdict
import numpy as np
from typing import List, Optional
from PIL import Image
from diffusers import FluxTransformer2DModel, AutoencoderKL
from contextlib import contextmanager
from safetensors.torch import save_file


class FSDP_EMA:
    """Exponential Moving Average handler for FSDP models."""
    
    def __init__(self, model, decay, rank):
        self.decay = decay
        self.rank = rank
        self.ema_state_dict_rank0 = {}
        
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(model, options=options)

        if self.rank == 0:
            self.ema_state_dict_rank0 = {k: v.clone() for k, v in state_dict.items()}
            main_print("--> Modern EMA handler initialized on rank 0.")

    def update(self, model):
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state_dict = get_model_state_dict(model, options=options)

        if self.rank == 0:
            for key in self.ema_state_dict_rank0:
                if key in model_state_dict:
                    self.ema_state_dict_rank0[key].copy_(
                        self.decay * self.ema_state_dict_rank0[key] + (1 - self.decay) * model_state_dict[key]
                    )

    @contextmanager
    def use_ema_weights(self, model):
        backup_options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        backup_state_dict_rank0 = get_model_state_dict(model, options=backup_options)

        load_options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
        set_model_state_dict(
            model,
            model_state_dict=self.ema_state_dict_rank0,
            options=load_options
        )
        
        try:
            yield
        finally:
            restore_options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
            set_model_state_dict(
                model,
                model_state_dict=backup_state_dict_rank0,
                options=restore_options
            )


def save_ema_checkpoint(ema_handler, rank, output_dir, step, epoch, config_dict):
    """Save EMA checkpoint."""
    if rank == 0:
        save_dir = os.path.join(output_dir, f"ema_checkpoint_{step}")
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, "ema_state_dict.safetensors")
        save_file(ema_handler.ema_state_dict_rank0, ckpt_path)
        main_print(f"--> EMA checkpoint saved to {save_dir}")


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    """Pack latents into the format expected by Flux."""
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def unpack_latents(latents, height, width, vae_scale_factor):
    """Unpack latents from Flux format."""
    batch_size, num_patches, channels = latents.shape
    height = height // vae_scale_factor
    width = width // vae_scale_factor
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, height, width)
    return latents


def prepare_latent_image_ids(height, width, device, dtype):
    """Prepare image position IDs for Flux Kontext.
    
    Returns 2D tensor (seq_len, 3) as expected by FluxTransformer2DModel.
    The first dimension (ids[..., 0]) is 0 for target latents.
    """
    latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = torch.arange(height, device=device, dtype=dtype)[:, None]
    latent_image_ids[..., 2] = torch.arange(width, device=device, dtype=dtype)[None, :]
    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    return latent_image_ids


def prepare_source_image_ids(height, width, device, dtype):
    """Prepare source image position IDs for Flux Kontext.
    
    Returns 2D tensor (seq_len, 3) with first dimension set to 1
    to distinguish source image from target latents.
    """
    source_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
    source_image_ids[..., 0] = 1  # Mark as source image (Kontext specific)
    source_image_ids[..., 1] = torch.arange(height, device=device, dtype=dtype)[:, None]
    source_image_ids[..., 2] = torch.arange(width, device=device, dtype=dtype)[None, :]
    source_image_ids = source_image_ids.reshape(height * width, 3)
    return source_image_ids


def sd3_time_shift(shift, t):
    """Apply time shift for SD3/Flux models."""
    return shift * t / (1 + (shift - 1) * t)


def flux_step(pred, latents, eta, sigma_schedule, step, prev_sample=None, grpo=False, sde_solver=False):
    """Single Flux sampling step with log probability calculation."""
    sigma_curr = sigma_schedule[step]
    sigma_next = sigma_schedule[step + 1]
    
    if sde_solver:
        dt = sigma_next - sigma_curr
        dw = torch.randn_like(latents) * ((-dt).sqrt())
        
        pred_original = latents - sigma_curr * pred
        next_sample = latents + pred * dt + eta * dw
        
        noise = (next_sample - latents - pred * dt) / (eta + 1e-8)
        log_prob = -0.5 * (noise ** 2).sum(dim=tuple(range(1, noise.ndim)))
    else:
        dt = sigma_next - sigma_curr
        pred_original = latents - sigma_curr * pred
        next_sample = latents + pred * dt
        log_prob = torch.zeros(latents.shape[0], device=latents.device)
    
    if grpo:
        return next_sample, pred_original, log_prob
    return next_sample


def run_sample_step_kontext(
    args,
    input_latents,
    progress_bar,
    sigma_schedule,
    transformer,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    image_ids,
    source_latents,
    source_image_ids,
    grpo_sample=False,
):
    """Run sampling steps for Kontext model."""
    latents = input_latents
    batch_latents = [latents.clone()]
    batch_log_probs = []
    
    for i in progress_bar:
        timestep_value = int(sigma_schedule[i] * 1000)
        timestep = torch.tensor([timestep_value], device=latents.device, dtype=torch.long)
        timestep = timestep.expand(latents.shape[0])
        
        # Concatenate target latents with source latents for Kontext conditioning
        combined_latents = torch.cat([latents, source_latents], dim=1)
        combined_image_ids = torch.cat([image_ids, source_image_ids], dim=0)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = transformer(
                hidden_states=combined_latents,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep / 1000,
                guidance=torch.tensor([3.5], device=latents.device, dtype=torch.bfloat16),
                img_ids=combined_image_ids,
                txt_ids=text_ids,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
        
        # Extract the target part (first half)
        pred = pred[:, :latents.shape[1], :]
        
        next_sample, _, log_prob = flux_step(
            pred, latents.to(torch.float32), args.eta, sigma_schedule, i, grpo=True, sde_solver=True
        )
        
        latents = next_sample.to(torch.bfloat16)
        batch_latents.append(latents.clone())
        batch_log_probs.append(log_prob)
    
    batch_latents = torch.stack(batch_latents, dim=1)
    batch_log_probs = torch.stack(batch_log_probs, dim=1)
    
    return batch_latents, batch_log_probs


def sample_single_image(
    args,
    device,
    transformer,
    vae,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    source_latents,
    caption,
    image_id,
    reward_model,
    image_processor,
    sample_idx,
    local_rank,
):
    """Sample a single image and return its data.
    
    Args:
        image_id: Image ID for CTR metadata lookup
    
    Returns:
        dict containing latents, log_probs, decoded image, reward, and all conditioning data
    """
    w, h = args.w, args.h
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)
    
    SPATIAL_DOWNSAMPLE = 8
    IN_CHANNELS = 16
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE
    
    # Get batch data (single sample)
    batch_encoder_hidden_states = encoder_hidden_states.unsqueeze(0).to(device)
    batch_pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0).to(device)
    batch_text_ids = text_ids.unsqueeze(0).to(device) if text_ids.dim() == 2 else text_ids.to(device)
    batch_source_latents = source_latents.unsqueeze(0).to(device) if source_latents is not None else None
    
    # Pack source latents to Flux format
    if batch_source_latents is not None:
        source_latents_packed = pack_latents(
            batch_source_latents, 1, IN_CHANNELS, latent_h, latent_w
        )
    else:
        raise ValueError("Source latents must be pre-computed for Kontext training")
    
    # Initialize noise
    noise = torch.randn(
        (1, IN_CHANNELS, latent_h, latent_w),
        device=device,
        dtype=torch.bfloat16,
    )
    latents = pack_latents(noise, 1, IN_CHANNELS, latent_h, latent_w)
    
    # Prepare image IDs
    image_ids = prepare_latent_image_ids(latent_h // 2, latent_w // 2, device, torch.bfloat16)
    source_image_ids = prepare_source_image_ids(latent_h // 2, latent_w // 2, device, torch.bfloat16)
    
    # Prepare text_ids with correct shape
    if batch_text_ids.dim() == 3:
        sample_text_ids = batch_text_ids[0]
    else:
        sample_text_ids = batch_text_ids
    
    # Run sampling
    progress_bar = range(sample_steps)
    batch_latents, batch_log_probs = run_sample_step_kontext(
        args,
        latents,
        progress_bar,
        sigma_schedule.to(device),
        transformer,
        batch_encoder_hidden_states,
        batch_pooled_prompt_embeds,
        sample_text_ids,
        image_ids,
        source_latents_packed,
        source_image_ids,
        grpo_sample=True,
    )
    
    # Decode final latents
    final_latents = batch_latents[:, -1]
    final_latents_unpacked = unpack_latents(final_latents, h, w, SPATIAL_DOWNSAMPLE)
    
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            final_latents_unpacked = (final_latents_unpacked / 0.3611) + 0.1159
            decoded_images = vae.decode(final_latents_unpacked, return_dict=False)[0]
    
    # Convert to PIL
    img_tensor = decoded_images[0]
    pil_img = image_processor.postprocess(img_tensor.unsqueeze(0))[0]
    
    # Save image for debugging
    image_path = os.path.join(
        "./images",
        f"flux_kontext_dpo_sample_{sample_idx}_rank_{local_rank}.jpg"
    )
    os.makedirs("./images", exist_ok=True)
    pil_img.save(image_path)
    
    # Compute reward with image_id for CTR metadata lookup
    score = reward_model.compute_reward(pil_img, caption, image_id=image_id)
    
    return {
        "latents": batch_latents,  # [1, T, seq_len, features]
        "log_probs": batch_log_probs,  # [1, T]
        "image_ids": image_ids.unsqueeze(0),  # [1, seq_len, 3]
        "source_latents": source_latents_packed,  # [1, seq_len, features]
        "encoder_hidden_states": batch_encoder_hidden_states,
        "pooled_prompt_embeds": batch_pooled_prompt_embeds,
        "text_ids": sample_text_ids.unsqueeze(0) if sample_text_ids.dim() == 2 else batch_text_ids,
        "pil_image": pil_img,
        "reward": score,
        "sigma_schedule": sigma_schedule,
    }


def sample_with_filter(
    args,
    device,
    transformer,
    vae,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    source_latents,
    caption,
    image_id,
    reward_model,
    image_filter: QwenImageFilter,
    image_processor,
    prompt_idx,
    local_rank,
    num_valid_required: int = 2,
    max_attempts: int = 20,
):
    """Sample images until we get the required number of valid samples.
    
    Args:
        image_id: Image ID for CTR metadata lookup
        num_valid_required: Number of valid samples needed (default 2 for DPO pairs)
        max_attempts: Maximum sampling attempts before giving up
        image_filter: QwenImageFilter instance for quality validation (required)
        
    Returns:
        List of valid sample dicts, each containing latents, rewards, etc.
    """
    valid_samples = []
    attempt = 0
    
    while len(valid_samples) < num_valid_required and attempt < max_attempts:
        # Sample a single image
        sample_data = sample_single_image(
            args,
            device,
            transformer,
            vae,
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids,
            source_latents,
            caption,
            image_id,
            reward_model,
            image_processor,
            sample_idx=f"{prompt_idx}_{attempt}",
            local_rank=local_rank,
        )
        
        attempt += 1
        
        # Check if image passes quality filter (required)
        is_valid = image_filter.validate(sample_data["pil_image"])
        if not is_valid:
            main_print(f"  Sample {attempt} rejected by Qwen filter")
            continue
        
        valid_samples.append(sample_data)
        main_print(f"  Valid sample {len(valid_samples)}/{num_valid_required} (attempt {attempt})")
    
    if len(valid_samples) < num_valid_required:
        main_print(f"Warning: Only got {len(valid_samples)} valid samples after {max_attempts} attempts")
    
    return valid_samples


def sample_reference_model_with_filter(
    args,
    device,
    transformer,
    vae,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    source_latents,
    captions,
    image_ids_list,
    reward_model,
    image_filter: QwenImageFilter,
    num_valid_per_prompt: int = 2,
):
    """Sample from reference model with Qwen3-VL quality filtering.
    
    For each prompt, keeps sampling until we get num_valid_per_prompt valid images
    that pass the Qwen3-VL quality check.
    
    Args:
        image_ids_list: List of image IDs for CTR metadata lookup
        image_filter: QwenImageFilter instance for quality validation (required)
        num_valid_per_prompt: Number of valid samples to collect per prompt (default 2)
        
    Returns:
        Same format as sample_reference_model, but with only valid samples
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    B = encoder_hidden_states.shape[0]
    
    all_latents = []
    all_log_probs = []
    all_rewards = []
    all_image_ids = []
    all_encoder_hidden_states = []
    all_pooled_prompt_embeds = []
    all_text_ids = []
    all_source_latents = []
    
    vae.enable_tiling()
    image_processor = VaeImageProcessor(16)
    
    # Sample for each prompt
    for prompt_idx in range(B):
        main_print(f"Sampling for prompt {prompt_idx + 1}/{B}")
        
        valid_samples = sample_with_filter(
            args,
            device,
            transformer,
            vae,
            encoder_hidden_states[prompt_idx],
            pooled_prompt_embeds[prompt_idx],
            text_ids[prompt_idx],
            source_latents[prompt_idx] if source_latents is not None else None,
            captions[prompt_idx],
            image_ids_list[prompt_idx],  # Pass image_id for CTR metadata lookup
            reward_model,
            image_filter,
            image_processor,
            prompt_idx=prompt_idx,
            local_rank=local_rank,
            num_valid_required=num_valid_per_prompt,
            max_attempts=args.max_sample_attempts,
        )
        
        # Collect valid samples
        for sample in valid_samples:
            all_latents.append(sample["latents"])
            all_log_probs.append(sample["log_probs"])
            all_rewards.append(sample["reward"])
            all_image_ids.append(sample["image_ids"])
            all_encoder_hidden_states.append(sample["encoder_hidden_states"])
            all_pooled_prompt_embeds.append(sample["pooled_prompt_embeds"])
            all_text_ids.append(sample["text_ids"])
            all_source_latents.append(sample["source_latents"])
    
    # Get sigma_schedule from last sample (all the same)
    sigma_schedule = valid_samples[-1]["sigma_schedule"] if valid_samples else torch.linspace(1, 0, args.sampling_steps + 1)
    
    # Stack all data
    all_rewards = torch.tensor(all_rewards, device=device, dtype=torch.float32)
    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_image_ids = torch.cat(all_image_ids, dim=0)
    all_source_latents = torch.cat(all_source_latents, dim=0)
    all_encoder_hidden_states = torch.cat(all_encoder_hidden_states, dim=0)
    all_pooled_prompt_embeds = torch.cat(all_pooled_prompt_embeds, dim=0)
    all_text_ids = torch.cat(all_text_ids, dim=0)
    
    return (
        all_rewards, all_latents, all_log_probs, sigma_schedule,
        all_image_ids, all_source_latents,
        all_encoder_hidden_states, all_pooled_prompt_embeds, all_text_ids
    )


def sample_reference_model(
    args,
    device,
    transformer,
    vae,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    source_latents,
    captions,
    reward_model,
):
    """Sample from reference model and compute rewards for Kontext."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    w, h = args.w, args.h
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    B = encoder_hidden_states.shape[0]
    SPATIAL_DOWNSAMPLE = 8
    IN_CHANNELS = 16
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    batch_size = 1
    batch_indices = torch.chunk(torch.arange(B), max(1, B // batch_size))

    all_latents = []
    all_log_probs = []
    all_raw_scores = []
    all_decoded_images = []
    all_image_ids = []
    all_encoder_hidden_states = []
    all_pooled_prompt_embeds = []
    all_text_ids = []
    all_source_latents = []

    vae.enable_tiling()
    image_processor = VaeImageProcessor(16)

    for index, batch_idx in enumerate(batch_indices):
        current_batch_size = len(batch_idx)
        
        # Get batch data
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx].to(device)
        batch_pooled_prompt_embeds = pooled_prompt_embeds[batch_idx].to(device)
        batch_text_ids = text_ids[batch_idx].to(device)
        batch_source_latents = source_latents[batch_idx].to(device) if source_latents is not None else None
        
        # Pack source latents to Flux format
        if batch_source_latents is not None:
            source_latents_packed = pack_latents(
                batch_source_latents, current_batch_size, IN_CHANNELS, latent_h, latent_w
            )
        else:
            raise ValueError("Source latents must be pre-computed for Kontext training")
        
        # Initialize noise
        noise = torch.randn(
            (current_batch_size, IN_CHANNELS, latent_h, latent_w),
            device=device,
            dtype=torch.bfloat16,
        )
        latents = pack_latents(noise, current_batch_size, IN_CHANNELS, latent_h, latent_w)
        
        # Prepare image IDs
        image_ids = prepare_latent_image_ids(latent_h // 2, latent_w // 2, device, torch.bfloat16)
        source_image_ids = prepare_source_image_ids(latent_h // 2, latent_w // 2, device, torch.bfloat16)
        
        # Prepare text_ids with correct shape
        if batch_text_ids.dim() == 3:
            sample_text_ids = batch_text_ids[0]
        else:
            sample_text_ids = batch_text_ids
        
        # Run sampling
        progress_bar = range(sample_steps)
        batch_latents, batch_log_probs = run_sample_step_kontext(
            args,
            latents,
            progress_bar,
            sigma_schedule.to(device),
            transformer,
            batch_encoder_hidden_states,
            batch_pooled_prompt_embeds,
            sample_text_ids,
            image_ids,
            source_latents_packed,
            source_image_ids,
            grpo_sample=True,
        )
        
        # Decode final latents
        final_latents = batch_latents[:, -1]
        final_latents_unpacked = unpack_latents(final_latents, h, w, SPATIAL_DOWNSAMPLE)
        
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                final_latents_unpacked = (final_latents_unpacked / 0.3611) + 0.1159
                decoded_images = vae.decode(final_latents_unpacked, return_dict=False)[0]
        
        # Convert to PIL and compute rewards
        for i in range(current_batch_size):
            img_tensor = decoded_images[i]
            pil_img = image_processor.postprocess(img_tensor.unsqueeze(0))[0]
            
            # Save image for debugging
            image_path = os.path.join(
                "./images",
                f"flux_kontext_dpo_{index}_{i}_rank_{local_rank}.jpg"
            )
            os.makedirs("./images", exist_ok=True)
            pil_img.save(image_path)
            
            all_decoded_images.append(pil_img)
            
            # Compute reward
            caption = captions[batch_idx[i]]
            score = reward_model.compute_reward(pil_img, caption)
            all_raw_scores.append({"reward": score})
        
        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)
        all_image_ids.append(image_ids.unsqueeze(0).expand(current_batch_size, -1, -1))
        all_encoder_hidden_states.append(batch_encoder_hidden_states)
        all_pooled_prompt_embeds.append(batch_pooled_prompt_embeds)
        all_text_ids.append(sample_text_ids.unsqueeze(0) if sample_text_ids.dim() == 2 else batch_text_ids)
        all_source_latents.append(source_latents_packed)
    
    # Compute final rewards
    all_rewards = torch.tensor(
        [s.get("reward", 0.0) for s in all_raw_scores], 
        device=device, 
        dtype=torch.float32
    )
    
    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_image_ids = torch.cat(all_image_ids, dim=0)
    all_source_latents = torch.cat(all_source_latents, dim=0)
    all_encoder_hidden_states = torch.cat(all_encoder_hidden_states, dim=0)
    all_pooled_prompt_embeds = torch.cat(all_pooled_prompt_embeds, dim=0)
    all_text_ids = torch.cat(all_text_ids, dim=0)
    
    return (
        all_rewards, all_latents, all_log_probs, sigma_schedule,
        all_image_ids, all_source_latents,
        all_encoder_hidden_states, all_pooled_prompt_embeds, all_text_ids
    )


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def dpo_one_step_kontext(
    args,
    transformer,
    latents_w,
    latents_l,
    sigma_schedule,
    train_timestep_idx,
    encoder_hidden_states,
    pooled_prompt_embeds,
    text_ids,
    image_ids,
    source_latents,
):
    """Single DPO training step for Kontext model.
    
    Args:
        latents_w: Winner latents [B, T, seq_len, features]
        latents_l: Loser latents [B, T, seq_len, features]
    
    Returns:
        theta_mse_w, theta_mse_l: MSE errors for winner and loser
    """
    batch_size = latents_w.shape[0]
    
    # Get final latents (target)
    target_w = latents_w[:, -1]  # [B, seq_len, features]
    target_l = latents_l[:, -1]
    
    # Sample random timesteps using logit-normal distribution
    u = compute_density_for_timestep_sampling(
        weighting_scheme='logit_normal',
        batch_size=batch_size,
        logit_mean=0,
        logit_std=1,
        mode_scale=1.29,
    )
    
    # Convert to timestep indices
    indices = (u * args.sampling_steps).long().clamp(0, args.sampling_steps - 1)
    sigmas = sigma_schedule[indices].to(latents_w.device)
    
    # Generate noise (same for winner and loser as in Diffusion-DPO)
    noise = torch.randn_like(target_w)
    
    # Add noise to latents according to flow matching: x_t = (1 - sigma) * x_0 + sigma * noise
    sigmas_expanded = sigmas.view(-1, 1, 1)
    noisy_latents_w = (1.0 - sigmas_expanded) * target_w + sigmas_expanded * noise
    noisy_latents_l = (1.0 - sigmas_expanded) * target_l + sigmas_expanded * noise
    
    # Stack for batch processing
    noisy_latents = torch.cat([noisy_latents_w, noisy_latents_l], dim=0)
    timesteps = torch.cat([sigmas, sigmas], dim=0) * 1000
    
    # Expand conditioning for both winner and loser
    encoder_hidden_states_exp = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=0)
    pooled_prompt_embeds_exp = torch.cat([pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    text_ids_exp = torch.cat([text_ids, text_ids], dim=0)
    image_ids_exp = torch.cat([image_ids, image_ids], dim=0)
    source_latents_exp = torch.cat([source_latents, source_latents], dim=0)
    
    # Prepare image IDs for Kontext
    # image_ids_exp shape: [2B, seq_len, 3] - target latent ids (first dim = 0)
    image_ids_input = image_ids_exp[0] if image_ids_exp.dim() == 3 else image_ids_exp
    source_image_ids = image_ids_input.clone()
    source_image_ids[..., 0] = 1  # Mark as source image
    
    # Concatenate target and source latents for Kontext
    combined_latents = torch.cat([noisy_latents, source_latents_exp], dim=1)
    combined_image_ids = torch.cat([image_ids_input, source_image_ids], dim=0)
    
    text_ids_input = text_ids_exp[0] if text_ids_exp.dim() == 3 else text_ids_exp
    
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = transformer(
            hidden_states=combined_latents,
            encoder_hidden_states=encoder_hidden_states_exp,
            timestep=timesteps / 1000,
            guidance=torch.tensor([3.5], device=noisy_latents.device, dtype=torch.bfloat16),
            img_ids=combined_image_ids,
            txt_ids=text_ids_input,
            pooled_projections=pooled_prompt_embeds_exp,
            return_dict=False,
        )[0]
    
    # Extract the target part (first half)
    pred = pred[:, :noisy_latents.shape[1], :]
    
    # Target is: noise - latent (velocity target for flow matching)
    noise_exp = torch.cat([noise, noise], dim=0)
    target_exp = torch.cat([target_w, target_l], dim=0)
    velocity_target = noise_exp - target_exp
    
    # Compute MSE
    mse = ((pred.float() - velocity_target.float()) ** 2).reshape(pred.shape[0], -1).mean(dim=1)
    
    theta_mse_w = mse[:batch_size]
    theta_mse_l = mse[batch_size:]
    
    return theta_mse_w, theta_mse_l


def train_one_step_dpo(
    args,
    device,
    transformer,
    transformer_ref,
    vae,
    reward_model,
    optimizer,
    lr_scheduler,
    batch_data,
    max_grad_norm,
    ema_handler,
    image_filter: QwenImageFilter,
):
    """Single DPO training step for Kontext model using pre-computed embeddings.
    
    Args:
        image_filter: QwenImageFilter for quality filtering (required).
    """
    total_loss = 0.0
    optimizer.zero_grad()
    
    # Unpack batch data (now includes image_ids instead of t2i_prompts)
    (
        encoder_hidden_states,
        pooled_prompt_embeds,
        text_ids,
        source_latents,
        captions,
        image_ids_list,  # Image IDs for CTR metadata lookup
    ) = batch_data
    
    B = encoder_hidden_states.shape[0]
    
    # Sample with Qwen3-VL filtering (required)
    # Collect exactly num_valid_samples valid samples per prompt
    (
        reward, all_latents, all_log_probs, sigma_schedule,
        all_image_ids, all_source_latents,
        all_encoder_hidden_states, all_pooled_prompt_embeds, all_text_ids
    ) = sample_reference_model_with_filter(
        args,
        device,
        transformer_ref if transformer_ref is not None else transformer,
        vae,
        encoder_hidden_states,
        pooled_prompt_embeds,
        text_ids,
        source_latents,
        captions,
        image_ids_list,  # Pass image_ids for CTR metadata lookup
        reward_model,
        image_filter=image_filter,
        num_valid_per_prompt=args.num_valid_samples,
    )
    
    # With filtering, we have exactly num_valid_samples per prompt
    n_prompts = B
    n_gens = args.num_valid_samples
    
    gathered_reward = gather_tensor(reward)
    if dist.get_rank() == 0:
        main_print(f"gathered_reward mean: {gathered_reward.mean().item()}")
        with open('./reward_kontext_dpo.txt', 'a') as f:
            f.write(f"{gathered_reward.mean().item()}\n")
    
    # Prepare DPO pairs based on rewards
    rewards_np = reward.cpu().numpy()
    
    # Group latents by prompt and create winner/loser pairs
    dpo_latents_w = []
    dpo_latents_l = []
    dpo_encoder_hidden_states = []
    dpo_pooled_prompt_embeds = []
    dpo_text_ids = []
    dpo_image_ids = []
    dpo_source_latents = []
    
    for i in range(n_prompts):
        start_idx = i * n_gens
        end_idx = (i + 1) * n_gens
        group_rewards = rewards_np[start_idx:end_idx]
        
        # Skip if all rewards are the same
        if np.std(group_rewards) < 1e-6:
            continue
        
        # Get winner (highest reward) and loser (lowest reward)
        winner_idx = start_idx + np.argmax(group_rewards)
        loser_idx = start_idx + np.argmin(group_rewards)
        
        if winner_idx != loser_idx:
            dpo_latents_w.append(all_latents[winner_idx:winner_idx+1])
            dpo_latents_l.append(all_latents[loser_idx:loser_idx+1])
            dpo_encoder_hidden_states.append(all_encoder_hidden_states[winner_idx:winner_idx+1])
            dpo_pooled_prompt_embeds.append(all_pooled_prompt_embeds[winner_idx:winner_idx+1])
            dpo_text_ids.append(all_text_ids[winner_idx:winner_idx+1])
            dpo_image_ids.append(all_image_ids[winner_idx:winner_idx+1])
            dpo_source_latents.append(all_source_latents[winner_idx:winner_idx+1])
    
    if len(dpo_latents_w) == 0:
        main_print("No valid DPO pairs found, skipping this batch")
        return 0.0, 0.0, gathered_reward.mean().item()
    
    dpo_latents_w = torch.cat(dpo_latents_w, dim=0)
    dpo_latents_l = torch.cat(dpo_latents_l, dim=0)
    dpo_encoder_hidden_states = torch.cat(dpo_encoder_hidden_states, dim=0)
    dpo_pooled_prompt_embeds = torch.cat(dpo_pooled_prompt_embeds, dim=0)
    dpo_text_ids = torch.cat(dpo_text_ids, dim=0)
    dpo_image_ids = torch.cat(dpo_image_ids, dim=0)
    dpo_source_latents = torch.cat(dpo_source_latents, dim=0)
    
    num_dpo_pairs = dpo_latents_w.shape[0]
    sigma_schedule = sigma_schedule.to(device)
    
    info = defaultdict(list)
    grad_norm = 0.0
    
    # DPO Training
    for t_idx in range(int(args.timestep_fraction * args.sampling_steps)):
        perm = torch.randperm(num_dpo_pairs, device=device)
        
        for mini_batch_start in range(0, num_dpo_pairs, args.train_batch_size):
            mini_batch_end = min(mini_batch_start + args.train_batch_size, num_dpo_pairs)
            mini_batch_idx = perm[mini_batch_start:mini_batch_end].cpu()
            
            batch_latents_w = dpo_latents_w[mini_batch_idx]
            batch_latents_l = dpo_latents_l[mini_batch_idx]
            batch_encoder_hidden_states = dpo_encoder_hidden_states[mini_batch_idx]
            batch_pooled_prompt_embeds = dpo_pooled_prompt_embeds[mini_batch_idx]
            batch_text_ids = dpo_text_ids[mini_batch_idx]
            batch_image_ids = dpo_image_ids[mini_batch_idx]
            batch_source_latents = dpo_source_latents[mini_batch_idx]
            
            # Compute learner model errors
            theta_mse_w, theta_mse_l = dpo_one_step_kontext(
                args,
                transformer,
                batch_latents_w,
                batch_latents_l,
                sigma_schedule,
                t_idx,
                batch_encoder_hidden_states,
                batch_pooled_prompt_embeds,
                batch_text_ids,
                batch_image_ids,
                batch_source_latents,
            )
            
            # Compute reference model errors
            with torch.no_grad():
                ref_model = transformer_ref if transformer_ref is not None else transformer
                ref_mse_w, ref_mse_l = dpo_one_step_kontext(
                    args,
                    ref_model,
                    batch_latents_w,
                    batch_latents_l,
                    sigma_schedule,
                    t_idx,
                    batch_encoder_hidden_states,
                    batch_pooled_prompt_embeds,
                    batch_text_ids,
                    batch_image_ids,
                    batch_source_latents,
                )
            
            # DPO loss
            w_diff = theta_mse_w - ref_mse_w
            l_diff = theta_mse_l - ref_mse_l
            w_l_diff = w_diff - l_diff
            
            inside_term = -0.5 * args.beta * w_l_diff
            loss = -F.logsigmoid(inside_term)
            loss = torch.mean(loss)
            
            # Compute implicit accuracy
            implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
            
            loss = loss / (args.gradient_accumulation_steps * int(args.timestep_fraction * args.sampling_steps))
            loss.backward()
            
            total_loss += loss.item()
            
            info["loss"].append(loss.item() * args.gradient_accumulation_steps * int(args.timestep_fraction * args.sampling_steps))
            info["theta_mse_w"].append(theta_mse_w.mean().item())
            info["theta_mse_l"].append(theta_mse_l.mean().item())
            info["ref_mse_w"].append(ref_mse_w.mean().item())
            info["ref_mse_l"].append(ref_mse_l.mean().item())
            info["w_diff"].append(w_diff.mean().item())
            info["l_diff"].append(l_diff.mean().item())
            info["w_l_diff"].append(w_l_diff.mean().item())
            info["implicit_acc"].append(implicit_acc.item())
    
    grad_norm = transformer.clip_grad_norm_(max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    
    if ema_handler is not None:
        ema_handler.update(transformer)
    
    reward_mean = gathered_reward.mean().item()
    return total_loss, grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm, reward_mean, info


def main(args):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    initialize_sequence_parallel_state(args.sp_size)
    
    if args.seed is not None:
        set_seed(args.seed + rank)

    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs("./images", exist_ok=True)

    reward_model = get_reward_fn(args.reward_type, device, metadata_csv=args.ctr_metadata_csv)
    main_print(f"--> Using reward model: {args.reward_type}")
    if args.ctr_metadata_csv:
        main_print(f"--> Using CTR metadata from: {args.ctr_metadata_csv}")
    
    # Initialize Qwen3-VL image filter (required for MLLM-based quality assurance)
    qwen_filter_url = args.qwen_filter_url or "http://localhost:8171"
    image_filter = QwenImageFilter(qwen_filter_url)
    main_print(f"--> Using Qwen3-VL image filter at {qwen_filter_url}")
    main_print(f"--> Will sample until {args.num_valid_samples} valid images per prompt (max {args.max_sample_attempts} attempts)")

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    
    # Load learner transformer
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    
    # Load reference transformer (frozen)
    transformer_ref = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    transformer_ref.requires_grad_(False)

    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        False,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    
    transformer = FSDP(transformer, **fsdp_kwargs)
    transformer_ref = FSDP(transformer_ref, **fsdp_kwargs)

    ema_handler = None
    if args.use_ema:
        ema_handler = FSDP_EMA(transformer, args.ema_decay, rank)

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing
        )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ).to(device)

    main_print(f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}")
    main_print(f"--> model loaded (using pre-computed T5 embeddings)")
    main_print(f"--> DPO beta: {args.beta}")

    transformer.train()

    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = FluxKontextLatentDataset(
        args.data_json_path, 
        num_latent_t=args.num_latent_t,
        cfg_rate=args.cfg,
    )
    
    sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=flux_kontext_latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    if rank <= 0:
        swanlab.init(
            project="flux-kontext-online-dpo",
            config=vars(args),
        )

    total_batch_size = (
        world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    
    steps_per_epoch = len(train_dataloader)
    total_steps = min(args.max_train_steps, args.num_epochs * steps_per_epoch)
    
    main_print("***** Running Online DPO Training for Flux Kontext *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(f"  Total train batch size = {total_batch_size}")
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps = {total_steps}")
    main_print(f"  DPO beta = {args.beta}")

    step_times = deque(maxlen=100)
    global_step = init_steps

    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        
        for step, batch_data in enumerate(train_dataloader):
            step_start_time = time.time()
            
            if global_step >= args.max_train_steps:
                break
            
            # Update reference model periodically
            if global_step > 0 and global_step % args.ref_update_step == 0:
                main_print(f"Updating reference model at step {global_step}")
                # Copy learner weights to reference
                options = StateDictOptions(full_state_dict=True, cpu_offload=True)
                learner_state = get_model_state_dict(transformer, options=options)
                set_model_state_dict(
                    transformer_ref,
                    model_state_dict=learner_state,
                    options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
                )
            
            total_loss, grad_norm, reward_mean, info = train_one_step_dpo(
                args,
                device,
                transformer,
                transformer_ref,
                vae,
                reward_model,
                optimizer,
                lr_scheduler,
                batch_data,
                args.max_grad_norm,
                ema_handler,
                image_filter=image_filter,
            )
            
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
            remaining_steps = args.max_train_steps - global_step
            eta_seconds = remaining_steps * avg_step_time
            eta_hours = eta_seconds / 3600
            
            if rank <= 0:
                eta_str = f"{eta_hours:.1f}h" if eta_hours >= 1 else f"{eta_seconds/60:.1f}m"
                main_print(
                    f"[Step {global_step}/{args.max_train_steps}] "
                    f"loss={total_loss:.4f} | reward={reward_mean:.4f} | "
                    f"grad_norm={grad_norm:.4f} | step_time={step_time:.2f}s | ETA: {eta_str}"
                )
                
                # Log to SwanLab
                log_dict = {
                    "loss": np.mean(info["loss"]) if info["loss"] else 0.0,
                    "implicit_acc": np.mean(info["implicit_acc"]) if info["implicit_acc"] else 0.0,
                    "reward_mean": reward_mean,
                    "grad_norm": grad_norm,
                    "w_diff": np.mean(info["w_diff"]) if info["w_diff"] else 0.0,
                    "l_diff": np.mean(info["l_diff"]) if info["l_diff"] else 0.0,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                }
                swanlab.log(log_dict, step=global_step)
            
            # Save checkpoint
            if global_step > 0 and global_step % args.checkpointing_steps == 0:
                save_checkpoint(
                    transformer, rank, args.output_dir, global_step, epoch, vars(args)
                )
                if ema_handler is not None:
                    save_ema_checkpoint(ema_handler, rank, args.output_dir, global_step, epoch, vars(args))
            
            global_step += 1
        
        if global_step >= args.max_train_steps:
            break

    # Save final model
    if args.final_model_dir:
        save_final_model(
            transformer, rank, args.final_model_dir, epoch, vars(args)
        )
        if ema_handler is not None:
            save_ema_checkpoint(ema_handler, rank, args.final_model_dir, global_step, epoch, vars(args))
    else:
        main_print("Training completed. No final_model_dir specified, skipping final model save.")

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()
    
    if rank <= 0:
        swanlab.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_latent_t", type=int, default=1)
    
    # Model arguments
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--cfg", type=float, default=0.0)
    
    # Training arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs/flux_kontext_dpo")
    parser.add_argument("--final_model_dir", type=str, default=None,
                        help="Directory to save the final trained model.")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_warmup_steps", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--use_cpu_offload", action="store_true")
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--train_sp_batch_size", type=int, default=1)
    parser.add_argument("--fsdp_sharding_startegy", default="full")
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--master_weight_type", type=str, default="fp32")
    
    # Sampling arguments
    parser.add_argument("--h", type=int, default=1312)
    parser.add_argument("--w", type=int, default=784)
    parser.add_argument("--sampling_steps", type=int, default=16)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--sampler_seed", type=int, default=42)
    parser.add_argument("--shift", type=float, default=3.0)
    
    # DPO/GRPO arguments
    parser.add_argument("--use_group", action="store_true")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--init_same_noise", action="store_true")
    parser.add_argument("--timestep_fraction", type=float, default=0.6)
    parser.add_argument("--clip_range", type=float, default=1e-4)
    parser.add_argument("--adv_clip_max", type=float, default=5.0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--beta", type=float, default=5000.0,
                        help="DPO beta parameter for controlling preference strength")
    parser.add_argument("--ref_update_step", type=int, default=100,
                        help="Update reference model every N steps")
    
    # Qwen3-VL image filter arguments (required for MLLM-based quality assurance)
    parser.add_argument("--qwen_filter_url", type=str, default="http://localhost:8171",
                        help="URL of the Qwen3-VL filter API server (required)")
    parser.add_argument("--num_valid_samples", type=int, default=2,
                        help="Number of valid samples to collect per prompt (default 2 for DPO pairs)")
    parser.add_argument("--max_sample_attempts", type=int, default=20,
                        help="Maximum sampling attempts per prompt before giving up")
    
    # Reward arguments
    parser.add_argument("--reward_type", type=str, default="hps_ctr",
                        choices=["hps", "hpsv2", "ctr", "hps_ctr"])
    parser.add_argument("--ctr_metadata_csv", type=str, required=True,
                        help="Path to CSV file containing CTR metadata (image_id, food_name, food_type, city_name, shop_name). Required for CTR reward lookup.")
    
    args = parser.parse_args()
    main(args)
