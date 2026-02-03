# Copyright (c) [2025] [FastVideo Team]
# SPDX-License-Identifier: [Apache License 2.0]
#
# Dataset for Flux Kontext instruction editing task.
# Data format: {"tag": "...", "include": [...], "exclude": [...], 
#               "t2i_prompt": "...", "prompt": "...", "image": "..."}

import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import numpy as np


class FluxKontextDataset(Dataset):
    """Dataset for Flux Kontext instruction-based image editing.
    
    Expected JSONL format:
    {
        "tag": "counting", 
        "include": [{"class": "airplane", "count": 2}], 
        "exclude": [{"class": "airplane", "count": 3}], 
        "t2i_prompt": "a photo of one airplanes", 
        "prompt": "Change the number of airplane in the image to two.", 
        "image": "generated_images/image_66.jpg"
    }
    """
    
    def __init__(
        self,
        jsonl_path: str,
        image_size: int = 512,
        cfg_rate: float = 0.0,
    ):
        self.jsonl_path = jsonl_path
        self.image_size = image_size
        self.cfg_rate = cfg_rate
        self.data_dir = os.path.dirname(jsonl_path)
        
        # Load data from JSONL
        self.data_anno = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data_anno.append(json.loads(line))
        
        # Unconditioned embeddings placeholder
        self.uncond_prompt_embed = torch.zeros(256, 4096).to(torch.float32)
        
        self.lengths = [1 for _ in self.data_anno]
    
    def __len__(self):
        return len(self.data_anno)
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image."""
        full_path = os.path.join(self.data_dir, image_path)
        image = Image.open(full_path).convert('RGB')
        
        # Resize to target size while maintaining aspect ratio
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        return image
    
    def __getitem__(self, idx):
        item = self.data_anno[idx]
        
        # Get editing instruction prompt
        edit_prompt = item['prompt']
        
        # Get original image path
        image_path = item['image']
        
        # Get additional metadata
        tag = item.get('tag', '')
        t2i_prompt = item.get('t2i_prompt', '')
        
        # Load source image
        source_image = self._load_image(image_path)
        
        return {
            'edit_prompt': edit_prompt,
            't2i_prompt': t2i_prompt,
            'tag': tag,
            'source_image': source_image,
            'image_path': image_path,
        }


def flux_kontext_collate_function(batch):
    """Collate function for Flux Kontext dataset."""
    edit_prompts = [item['edit_prompt'] for item in batch]
    t2i_prompts = [item['t2i_prompt'] for item in batch]
    tags = [item['tag'] for item in batch]
    source_images = [item['source_image'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'edit_prompts': edit_prompts,
        't2i_prompts': t2i_prompts,
        'tags': tags,
        'source_images': source_images,
        'image_paths': image_paths,
    }


class FluxKontextLatentDataset(Dataset):
    """Dataset with pre-computed latents and embeddings for Flux Kontext.
    
    This dataset expects pre-processed embeddings stored in subdirectories.
    """
    
    def __init__(
        self,
        json_path: str,
        num_latent_t: int = 1,
        cfg_rate: float = 0.0,
    ):
        self.json_path = json_path
        self.cfg_rate = cfg_rate
        self.data_dir = os.path.dirname(json_path)
        
        # Directories for pre-computed embeddings
        self.prompt_embed_dir = os.path.join(self.data_dir, "prompt_embed")
        self.pooled_prompt_embeds_dir = os.path.join(self.data_dir, "pooled_prompt_embeds")
        self.text_ids_dir = os.path.join(self.data_dir, "text_ids")
        self.source_latents_dir = os.path.join(self.data_dir, "source_latents")
        
        # Load data annotations
        with open(json_path, 'r') as f:
            self.data_anno = json.load(f)
        
        self.num_latent_t = num_latent_t
        
        # Unconditioned embeddings
        self.uncond_prompt_embed = torch.zeros(256, 4096).to(torch.float32)
        self.uncond_prompt_mask = torch.zeros(256).bool()
        
        self.lengths = [
            data_item.get("length", 1) for data_item in self.data_anno
        ]
    
    def __len__(self):
        return len(self.data_anno)
    
    def __getitem__(self, idx):
        item = self.data_anno[idx]
        
        # Load prompt embeddings
        prompt_embed_file = item["prompt_embed_path"]
        pooled_prompt_embeds_file = item["pooled_prompt_embeds_path"]
        text_ids_file = item["text_ids"]
        source_latents_file = item.get("source_latents_path", None)
        
        prompt_embed = torch.load(
            os.path.join(self.prompt_embed_dir, prompt_embed_file),
            map_location="cpu",
            weights_only=True,
        )
        
        pooled_prompt_embeds = torch.load(
            os.path.join(self.pooled_prompt_embeds_dir, pooled_prompt_embeds_file),
            map_location="cpu",
            weights_only=True,
        )
        
        text_ids = torch.load(
            os.path.join(self.text_ids_dir, text_ids_file),
            map_location="cpu",
            weights_only=True,
        )
        
        # Load source image latents
        source_latents = None
        if source_latents_file:
            source_latents = torch.load(
                os.path.join(self.source_latents_dir, source_latents_file),
                map_location="cpu",
                weights_only=True,
            )
        
        caption = item['caption']  # This should be the edit prompt
        image_id = item.get('image_id', '')  # Image ID for CTR metadata lookup
        
        return {
            "prompt_embed": prompt_embed,
            "pooled_prompt_embed": pooled_prompt_embeds,
            "text_ids": text_ids,
            "source_latents": source_latents,
            "caption": caption,
            "image_id": image_id,
        }


def flux_kontext_latent_collate_function(batch):
    """Collate function for pre-computed latent dataset.
    
    Returns tuple to match train_one_step unpacking:
    (encoder_hidden_states, pooled_prompt_embeds, text_ids, source_latents, captions, image_ids)
    """
    prompt_embeds = [item["prompt_embed"] for item in batch]
    pooled_prompt_embeds = [item["pooled_prompt_embed"] for item in batch]
    text_ids = [item["text_ids"] for item in batch]
    source_latents = [item["source_latents"] for item in batch]
    captions = [item["caption"] for item in batch]
    image_ids = [item["image_id"] for item in batch]  # Image IDs for CTR metadata lookup
    
    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    pooled_prompt_embeds = torch.stack(pooled_prompt_embeds, dim=0)
    text_ids = torch.stack(text_ids, dim=0)
    
    # Handle source latents (may be None)
    if source_latents[0] is not None:
        source_latents = torch.stack(source_latents, dim=0)
    else:
        source_latents = None
    
    # Return tuple to match train_one_step unpacking
    return (
        prompt_embeds,       # encoder_hidden_states
        pooled_prompt_embeds,
        text_ids,
        source_latents,
        captions,
        image_ids,           # Image IDs for CTR metadata lookup
    )


if __name__ == "__main__":
    # Test the dataset
    dataset = FluxKontextDataset("data/test_metadata.jsonl", image_size=512)
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Edit prompt: {sample['edit_prompt']}")
        print(f"T2I prompt: {sample['t2i_prompt']}")
