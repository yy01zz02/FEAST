import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T



class ResizeWithPadding: 
    """Resize image while preserving aspect ratio with padding.
    
    Args:
        target_size: Target size (int or tuple of height, width)
        fill_color: RGB tuple for padding color (default: white)
    """
    def __init__(self, target_size, fill_color=(255, 255, 255)):
        if isinstance(target_size, int):
            self.target_h = self.target_w = target_size
        else:
            self.target_h, self.target_w = target_size
        self.fill_color = fill_color
    
    def __call__(self, img):
        orig_w, orig_h = img.size
        
        scale = min(self.target_w / orig_w, self.target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        new_img = Image.new('RGB', (self.target_w, self.target_h), self.fill_color)
        paste_x = (self.target_w - new_w) // 2
        paste_y = (self.target_h - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img


class FoodCTRDataset(Dataset):
    def __init__(self,
                 csv_path,
                 tokenizer,
                 image_processor,
                 max_seq_len=128,
                 target_size=384,
                 resize_mode='padding',
                 fill_color=(255, 255, 255)):
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_seq_len = max_seq_len
        self.target_size = target_size
        
        # Image resize transform
        if resize_mode == 'padding':
            self.resize_transform = ResizeWithPadding(target_size, fill_color)
        else:
            self.resize_transform = T.Resize((target_size, target_size))
        
        # Load CSV data
        self.data = pd.read_csv(csv_path)
        print(f"Raw data: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def _prepare_text(self, row):
        """Concatenate all text fields into a single description."""
        food_name = str(row.get('food_name', '')) if pd.notna(row.get('food_name')) else ''
        food_type = str(row.get('food_type', '')) if pd.notna(row.get('food_type')) else ''
        city_name = str(row.get('city_name', '')) if pd.notna(row.get('city_name')) else ''
        shop_name = str(row.get('shop_name', '')) if pd.notna(row.get('shop_name')) else ''
        
        # Format: "菜品:{food_name} 类型:{food_type} 城市:{city_name} 店铺:{shop_name}"
        text = f"菜品:{food_name} 类型:{food_type} 城市:{city_name} 店铺:{shop_name}"
        return text
    
    def _load_image(self, image_path):
        """Load and preprocess image from path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path)
        
        # Convert to RGB
        if image.mode == 'RGBA': 
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != 'RGB': 
            image = image.convert('RGB')
        
        # Apply resize transform
        image = self.resize_transform(image)
        
        # Apply image processor
        pixel_values = self.image_processor(
            images=image,
            do_resize=False,
            return_tensors="pt"
        ).pixel_values.squeeze(0)
        
        return pixel_values
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 1. Process text
        text = self._prepare_text(row)
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # 2. Load image
        pixel_values = self._load_image(row['image_url'])
        
        # 3. CTR label
        ctr = float(row['pv_ctr']) if pd.notna(row['pv_ctr']) else 0.0
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'ctr': torch.tensor([ctr], dtype=torch.float32),
            'image_url': str(row['image_url'])
        }

def collate_fn(batch):
    """Custom collate function for DataLoader."""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'ctr': torch.stack([item['ctr'] for item in batch]),
        'image_urls': [item['image_url'] for item in batch]
    }