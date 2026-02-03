# Copyright (c) [2025] [FastVideo Team]
# SPDX-License-Identifier: [Apache License 2.0]
#
# Unified reward functions for GRPO training.
# Reward functions are encapsulated here for easy switching between different reward models.
# To change reward model implementation, only modify this file.

import torch
import numpy as np
import requests
import base64
from io import BytesIO
from PIL import Image
from typing import Union, List, Optional


# ============================================================================
# Configuration - Modify these URLs if your reward API endpoints change
# ============================================================================
REWARD_API_CONFIG = {
    "hpsv2": {
        "url": "http://localhost:8163/score",
        "timeout": 30,
    },
    "ctr": {
        "url": "http://localhost:8199/predict",
        "timeout": 30,
    },
    "qwen_filter": {
        "url": "http://localhost:8171/validate",
        "timeout": 60,
    },
}


# ============================================================================
# Base Reward Class
# ============================================================================
class BaseReward:
    """Base class for reward functions."""
    
    def __init__(self, device):
        self.device = device
    
    def _encode_image(self, image: Union[Image.Image, np.ndarray, str]) -> str:
        """Encode image to base64 string."""
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _call_api(self, api_name: str, payload: dict) -> float:
        """Call reward API service."""
        config = REWARD_API_CONFIG.get(api_name)
        if not config:
            print(f"API config for {api_name} not found.")
            return 0.0
        
        try:
            resp = requests.post(
                config["url"], 
                json=payload, 
                timeout=config["timeout"]
            )
            resp.raise_for_status()
            return resp.json()['score']
        except requests.exceptions.Timeout:
            print(f"Timeout calling {api_name} API")
            return 0.0
        except requests.exceptions.RequestException as e:
            print(f"Error calling {api_name} API: {e}")
            return 0.0
        except Exception as e:
            print(f"Error calculating {api_name} score: {e}")
            return 0.0
    
    def compute_reward(
        self, 
        image: Union[Image.Image, np.ndarray, str], 
        prompt: str
    ) -> float:
        """Compute reward for a single image-prompt pair. Override in subclass."""
        raise NotImplementedError
    
    def __call__(
        self, 
        images: List[Union[Image.Image, np.ndarray, str]], 
        prompts: List[str]
    ) -> torch.Tensor:
        """Compute rewards for a batch of image-prompt pairs."""
        rewards = []
        for image, prompt in zip(images, prompts):
            score = self.compute_reward(image, prompt)
            rewards.append(score)
        return torch.tensor(rewards, device=self.device, dtype=torch.float32)


# ============================================================================
# HPS-only Reward Function
# ============================================================================
class HPSAPIReward(BaseReward):
    """HPSv2-based reward function.
    
    Computes Human Preference Score v2 for generated image.
    Uses API service for inference.
    """
    
    def __init__(self, device):
        super().__init__(device)
    
    def compute_reward(
        self, 
        image: Union[Image.Image, np.ndarray, str], 
        prompt: str
    ) -> float:
        """Compute HPSv2 score."""
        payload = {
            "image_base64": self._encode_image(image),
            "prompt": prompt
        }
        return self._call_api("hpsv2", payload)
    
    # Alias for backward compatibility
    def calculate_hpsv2_score(self, image, prompt):
        return self.compute_reward(image, prompt)


# ============================================================================
# CTR Metadata Manager (CSV-based)
# ============================================================================
class CTRMetadataManager:
    """Manages CTR metadata from a CSV file.
    
    CSV format expected:
        image_id,food_name,food_type,city_name,shop_name
        10001,红烧肉饭,中式快餐,上海,老王饭店
        10002,鱼香肉丝,川菜,北京,川味小馆
        ...
    
    The image_id can be extracted from image path/filename.
    """
    
    def __init__(self, csv_path: str = None):
        """Initialize metadata manager.
        
        Args:
            csv_path: Path to CSV file containing metadata.
                     If None, will use default values.
        """
        self.csv_path = csv_path
        self.metadata = {}
        self._loaded = False
        
        if csv_path is not None:
            self._load_csv(csv_path)
    
    def _load_csv(self, csv_path: str):
        """Load metadata from CSV file."""
        import csv
        import os
        
        if not os.path.exists(csv_path):
            print(f"Warning: CTR metadata CSV not found: {csv_path}")
            return
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Support both 'image_id' and 'id' as the key column
                    image_id = row.get('image_id') or row.get('id') or row.get('img_id')
                    if image_id:
                        self.metadata[str(image_id)] = {
                            'food_name': row.get('food_name', ''),
                            'food_type': row.get('food_type', ''),
                            'city_name': row.get('city_name', ''),
                            'shop_name': row.get('shop_name', ''),
                        }
            self._loaded = True
            print(f"Loaded {len(self.metadata)} metadata entries from {csv_path}")
        except Exception as e:
            print(f"Error loading CTR metadata CSV: {e}")
    
    def _extract_image_id(self, image_path: str) -> str:
        """Extract image ID from path or filename.
        
        Examples:
            /path/to/images/10001.jpg -> 10001
            /path/to/10001_edited.png -> 10001
            flux_kontext_dpo_0_1_rank_0.jpg -> 0_1
        """
        import os
        import re
        
        # Get filename without extension
        basename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(basename)[0]
        
        # Try to extract numeric ID first
        match = re.match(r'^(\d+)', name_without_ext)
        if match:
            return match.group(1)
        
        # Return the full name without extension as fallback
        return name_without_ext
    
    def get_metadata(self, image_id: str = None, image_path: str = None) -> dict:
        """Get metadata for an image.
        
        Args:
            image_id: Direct image ID to look up
            image_path: Image path to extract ID from
            
        Returns:
            Dict with keys: food_name, food_type, city_name, shop_name
            Returns empty values if not found.
        """
        # Determine the ID to look up
        lookup_id = image_id
        if lookup_id is None and image_path is not None:
            lookup_id = self._extract_image_id(image_path)
        
        if lookup_id is not None and str(lookup_id) in self.metadata:
            return self.metadata[str(lookup_id)]
        
        # Return empty dict if not found (no defaults)
        return {
            'food_name': '',
            'food_type': '',
            'city_name': '',
            'shop_name': '',
        }
    
    def is_loaded(self) -> bool:
        """Check if metadata was loaded successfully."""
        return self._loaded


# ============================================================================
# CTR Reward Function (Food CTR Prediction)
# ============================================================================
class CTRAPIReward(BaseReward):
    """CTR-based reward function for food image generation.
    
    Uses a CTR prediction model to score generated food images.
    Metadata (food_name, food_type, city_name, shop_name) is loaded from CSV file.
    
    Usage:
        # Initialize with metadata CSV
        reward = CTRAPIReward(device, metadata_csv="path/to/metadata.csv")
        
        # Compute reward (metadata looked up by image_id)
        score = reward.compute_reward(image, prompt, image_id="10001")
        
        # Or provide metadata directly
        score = reward.compute_reward(
            image, prompt, 
            food_name="红烧肉", food_type="中式快餐"
        )
    """
    
    def __init__(self, device, metadata_csv: str = None):
        super().__init__(device)
        
        # Initialize metadata manager
        self.metadata_manager = CTRMetadataManager(metadata_csv)
    
    def set_metadata_csv(self, csv_path: str):
        """Set or update the metadata CSV path.
        
        Args:
            csv_path: Path to CSV file containing metadata
        """
        self.metadata_manager = CTRMetadataManager(csv_path)
    
    def _call_ctr_api(
        self, 
        image: Union[Image.Image, np.ndarray, str],
        food_name: str,
        food_type: str,
        city_name: str,
        shop_name: str,
    ) -> float:
        """Call CTR API service using multipart form data."""
        config = REWARD_API_CONFIG.get("ctr")
        if not config:
            print("CTR API config not found.")
            return 0.0
        
        # Prepare image as file
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to bytes
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        
        try:
            files = {
                "image": ("image.png", buffered, "image/png")
            }
            data = {
                "food_name": food_name,
                "food_type": food_type,
                "city_name": city_name,
                "shop_name": shop_name,
            }
            
            resp = requests.post(
                config["url"],
                files=files,
                data=data,
                timeout=config["timeout"]
            )
            resp.raise_for_status()
            result = resp.json()
            
            if result.get("success"):
                return float(result["data"]["ctr"])
            else:
                print(f"CTR API error: {result.get('error', 'Unknown error')}")
                return 0.0
        except requests.exceptions.Timeout:
            print("Timeout calling CTR API")
            return 0.0
        except requests.exceptions.RequestException as e:
            print(f"Error calling CTR API: {e}")
            return 0.0
        except Exception as e:
            print(f"Error calculating CTR score: {e}")
            return 0.0
    
    def compute_reward(
        self, 
        image: Union[Image.Image, np.ndarray, str], 
        prompt: str,
        image_id: str = None,
        image_path: str = None,
        food_name: str = None,
        food_type: str = None,
        city_name: str = None,
        shop_name: str = None,
    ) -> float:
        """Compute CTR reward score.
        
        Args:
            image: Image to score
            prompt: Text prompt (not used for CTR, kept for interface compatibility)
            image_id: Image ID to look up metadata in CSV
            image_path: Image path to extract ID and look up metadata
            food_name: Food name (overrides CSV lookup)
            food_type: Food type (overrides CSV lookup)
            city_name: City name (overrides CSV lookup)
            shop_name: Shop name (overrides CSV lookup)
            
        Returns:
            CTR score (0-1)
        """
        # Get metadata from CSV if not provided directly
        if image_id is not None or image_path is not None:
            csv_metadata = self.metadata_manager.get_metadata(
                image_id=image_id, 
                image_path=image_path
            )
        elif isinstance(image, str):
            # If image is a path, try to extract ID from it
            csv_metadata = self.metadata_manager.get_metadata(image_path=image)
        else:
            csv_metadata = {}
        
        # Use provided values or fall back to CSV values
        final_food_name = food_name or csv_metadata.get('food_name', '')
        final_food_type = food_type or csv_metadata.get('food_type', '')
        final_city_name = city_name or csv_metadata.get('city_name', '')
        final_shop_name = shop_name or csv_metadata.get('shop_name', '')
        
        return self._call_ctr_api(
            image, 
            final_food_name, 
            final_food_type, 
            final_city_name, 
            final_shop_name
        )


# ============================================================================
# HPS + CTR Hybrid Reward Function
# ============================================================================
class HPSCTRAPIReward(BaseReward):
    """Hybrid Reward combining CTR and Aesthetic scores.
    
    The final hybrid reward R combines the business-oriented CTR prediction ŷ 
    and the aesthetic score S_aes to guide the generation process.
    
    This implements the hybrid reward mechanism that:
    - Uses CTR (ŷ) to reflect business value
    - Uses HPS v2.1 as the aesthetic score (S_aes)
    - Combines both with configurable weights
    
    Default ratio is 1:1 (hps_weight=0.5, ctr_weight=0.5).
    
    Usage:
        # Initialize with metadata CSV
        reward = HPSCTRAPIReward(device, metadata_csv="path/to/metadata.csv")
        
        # Compute reward (metadata looked up by image_id)
        score = reward.compute_reward(image, prompt, image_id="10001")
    """
    
    def __init__(
        self, 
        device, 
        hps_weight: float = 0.5, 
        ctr_weight: float = 0.5,
        metadata_csv: str = None
    ):
        super().__init__(device)
        
        # Normalize weights
        total = hps_weight + ctr_weight
        self.hps_weight = hps_weight / total
        self.ctr_weight = ctr_weight / total
        
        # Initialize sub-reward functions
        self.hps_reward = HPSAPIReward(device)
        self.ctr_reward = CTRAPIReward(device, metadata_csv=metadata_csv)
    
    def set_metadata_csv(self, csv_path: str):
        """Set or update the metadata CSV path for CTR reward.
        
        Args:
            csv_path: Path to CSV file containing metadata
        """
        self.ctr_reward.set_metadata_csv(csv_path)
    
    def compute_reward(
        self, 
        image: Union[Image.Image, np.ndarray, str], 
        prompt: str,
        image_id: str = None,
        image_path: str = None,
        food_name: str = None,
        food_type: str = None,
        city_name: str = None,
        shop_name: str = None,
    ) -> float:
        """Compute combined HPS + CTR reward.
        
        HPS score is typically in range [0.15, 0.35], we normalize to [0, 1]
        CTR score is already in [0, 1]
        
        Args:
            image: Image to score
            prompt: Text prompt
            image_id: Image ID to look up metadata in CSV
            image_path: Image path to extract ID and look up metadata
            food_name: Food name (overrides CSV lookup)
            food_type: Food type (overrides CSV lookup)
            city_name: City name (overrides CSV lookup)
            shop_name: Shop name (overrides CSV lookup)
        """
        hps_score = self.hps_reward.compute_reward(image, prompt)
        ctr_score = self.ctr_reward.compute_reward(
            image, prompt, 
            image_id=image_id,
            image_path=image_path,
            food_name=food_name, 
            food_type=food_type, 
            city_name=city_name, 
            shop_name=shop_name
        )
        
        # Normalize HPS score to [0, 1] range (typical range is 0.15-0.35)
        hps_normalized = (hps_score - 0.15) / 0.2
        hps_normalized = max(0.0, min(1.0, hps_normalized))
        
        combined = self.hps_weight * hps_normalized + self.ctr_weight * ctr_score
        return combined


# ============================================================================
# Factory Function - Use this to get reward model by name
# ============================================================================
def get_reward_fn(reward_type: str, device, metadata_csv: str = None, **kwargs):
    """Get reward function by type name.
    
    Args:
        reward_type: One of 'hps', 'ctr', 'hps_ctr'
        device: Torch device
        metadata_csv: Path to CSV file containing CTR metadata (for ctr/hps_ctr types)
        **kwargs: Additional arguments passed to reward function constructor
        
    Returns:
        Reward function instance
        
    Example:
        reward_fn = get_reward_fn('hps', device)
        score = reward_fn.compute_reward(image, prompt)
        
        # Or for batch:
        scores = reward_fn(images, prompts)
        
        # For CTR with metadata CSV:
        reward_fn = get_reward_fn('ctr', device, metadata_csv="path/to/metadata.csv")
        score = reward_fn.compute_reward(image, prompt, image_id="10001")
        
        # For HPS+CTR with metadata CSV:
        reward_fn = get_reward_fn('hps_ctr', device, metadata_csv="path/to/metadata.csv")
        score = reward_fn.compute_reward(image, prompt, image_id="10001")
    """
    reward_types = {
        'hps': HPSAPIReward,
        'hps_api': HPSAPIReward,
        'hpsv2': HPSAPIReward,
        'ctr': CTRAPIReward,
        'ctr_api': CTRAPIReward,
        'hps_ctr': HPSCTRAPIReward,
        'hps_ctr_api': HPSCTRAPIReward,
        'hpsv2_ctr': HPSCTRAPIReward,
    }
    
    if reward_type not in reward_types:
        raise ValueError(
            f"Unknown reward type: {reward_type}. "
            f"Available types: {list(reward_types.keys())}"
        )
    
    reward_class = reward_types[reward_type]
    
    # For CTR-related rewards, pass metadata_csv if provided
    if reward_type in ['ctr', 'ctr_api', 'hps_ctr', 'hps_ctr_api', 'hpsv2_ctr']:
        return reward_class(device, metadata_csv=metadata_csv, **kwargs)
    else:
        return reward_class(device, **kwargs)


# Backward compatibility alias
create_reward_model = get_reward_fn
