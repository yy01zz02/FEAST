# Copyright (c) [2025] [FastVideo Team]
# SPDX-License-Identifier: [Apache License 2.0]
#
# Preprocessing script for Flux Kontext instruction editing.
# Generates T5 embeddings and VAE latents for training.

import argparse
import torch
import json
import os
import gc
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from diffusers import FluxPipeline
from diffusers.image_processor import VaeImageProcessor
from PIL import Image


class MetadataDataset(Dataset):
    """Dataset for loading metadata from JSONL file."""
    
    def __init__(self, jsonl_path):
        self.data_dir = os.path.dirname(jsonl_path)
        self.items = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.items.append(json.loads(line))
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        item['index'] = idx
        return item


def create_image_processor(vae_scale_factor: int = 8):
    """Create image processor matching official Flux pipeline.
    
    Flux uses vae_scale_factor * 2 for the image processor due to 2x2 patch packing.
    """
    image_processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor * 2,  # Flux uses 16 due to patch packing
        do_resize=True,
        do_normalize=True,  # Normalize to [-1, 1]
        do_convert_rgb=True,
    )
    return image_processor


def load_image_with_processor(
    image_path: str, 
    image_processor: VaeImageProcessor, 
    height: int = 512, 
    width: int = 512
) -> torch.Tensor:
    """Load and preprocess image using VaeImageProcessor.
    
    Returns tensor with shape [1, C, H, W] in range [-1, 1].
    """
    image = Image.open(image_path).convert('RGB')
    pixel_values = image_processor.preprocess(image, height=height, width=width)
    return pixel_values.float()


def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    subdirs = ["prompt_embed", "pooled_prompt_embeds", "text_ids", "source_latents"]
    for d in subdirs:
        os.makedirs(os.path.join(args.output_dir, d), exist_ok=True)

    # Load Pipeline (without transformer to save memory)
    print(f"Rank {local_rank}: Loading Pipeline...")
    pipe = FluxPipeline.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to(device)
    
    # We don't need the transformer for preprocessing
    pipe.transformer = None
    gc.collect()
    torch.cuda.empty_cache()
    
    # Create image processor
    vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    image_processor = create_image_processor(vae_scale_factor)
    print(f"Rank {local_rank}: VAE scale factor {vae_scale_factor}")

    dataset = MetadataDataset(args.jsonl_path)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        sampler=sampler, 
        num_workers=args.dataloader_num_workers,
        collate_fn=lambda x: x[0]
    )

    processed_items = []

    print(f"Rank {local_rank}: Starting processing...")
    for item in tqdm(dataloader, disable=local_rank != 0):
        try:
            filename_base = f"kontext_{item['index']}"
            
            # 1. Process Text Embeddings
            prompt = item['prompt']
            
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=prompt, 
                    prompt_2=prompt
                )
            
            # Save embeddings
            torch.save(prompt_embeds[0].cpu(), 
                      os.path.join(args.output_dir, "prompt_embed", f"{filename_base}.pt"))
            torch.save(pooled_prompt_embeds[0].cpu(), 
                      os.path.join(args.output_dir, "pooled_prompt_embeds", f"{filename_base}.pt"))
            torch.save(text_ids.cpu(), 
                      os.path.join(args.output_dir, "text_ids", f"{filename_base}.pt"))
            
            new_item = {
                "image_id": item.get("image_id", str(item['index'])),  # Image ID for CTR metadata lookup
                "caption": prompt,
                "prompt_embed_path": f"{filename_base}.pt",
                "pooled_prompt_embeds_path": f"{filename_base}.pt",
                "text_ids": f"{filename_base}.pt",
            }

            # 2. Process Source Image Latents
            image_path = os.path.join(dataset.data_dir, item['image'])
            pixel_values = load_image_with_processor(
                image_path, image_processor, args.height, args.width
            ).to(device)
            
            with torch.no_grad():
                pixel_values_bf16 = pixel_values.to(dtype=torch.bfloat16)
                latents = pipe.vae.encode(pixel_values_bf16).latent_dist.sample()
                latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            
            torch.save(latents[0].cpu(), 
                      os.path.join(args.output_dir, "source_latents", f"{filename_base}.pt"))
            
            new_item["source_latents_path"] = f"{filename_base}.pt"
            
            processed_items.append(new_item)

        except Exception as e:
            print(f"Rank {local_rank} Error processing {item['index']}: {e}")

    # Gather results from all ranks
    all_processed_items = [None] * world_size
    dist.all_gather_object(all_processed_items, processed_items)
    
    if local_rank == 0:
        flat_items = [item for sublist in all_processed_items for item in sublist]
        output_json = os.path.join(args.output_dir, "metadata.json")
        with open(output_json, 'w') as f:
            json.dump(flat_items, f, indent=4)
        print(f"Saved metadata to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for Flux Kontext training")
    parser.add_argument("--jsonl_path", type=str, required=True,
                       help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for preprocessed data")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to Flux model")
    parser.add_argument("--height", type=int, default=512,
                       help="Image height (must be divisible by 16)")
    parser.add_argument("--width", type=int, default=512,
                       help="Image width (must be divisible by 16)")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    
    args = parser.parse_args()
    main(args)