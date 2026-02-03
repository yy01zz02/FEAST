import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoImageProcessor

from model import FoodCTRModel
from dataset import FoodCTRDataset, collate_fn
import torch.nn.functional as F


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AverageMeter: 
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DistributionAwareCompoundLoss(nn.Module):
    """Distribution-aware Compound Loss.
    
    L_total = L_reg + λ1 * L_var + λ2 * KL(P_ŷ || P_y)
    
    This loss function addresses the "mean regression" trap caused by the
    long-tail distribution of food delivery data. It consists of three components:
    
    1. L_reg: Weighted regression loss with higher weight α for positive samples
    2. L_var: Variance penalty to prevent prediction collapse
    3. L_kl: KL divergence to align prediction distribution with ground truth
    
    Args:
        alpha: Weight for positive samples in L_reg (default: 10.0)
        lambda1: Coefficient for variance penalty L_var (default: 0.1)
        lambda2: Coefficient for KL divergence L_kl (default: 0.01)
    """
    def __init__(self, alpha=10.0, lambda1=0.1, lambda2=0.01):
        super().__init__()
        self.alpha = alpha      # Positive sample weight
        self.lambda1 = lambda1  # Variance penalty coefficient
        self.lambda2 = lambda2  # KL divergence coefficient
    
    def forward(self, pred, target):
        pred = pred.squeeze()  # Ensure 1D tensor
        target = target.squeeze()
        
        B = pred.size(0)
        
        # ==================== L_reg: Weighted Regression Loss ====================
        # L_reg = (1/B) * Σ w_i * (ŷ_i - y_i)²
        # where w_i = I(y_i > 0) * α + I(y_i = 0) * 1
        mse = (pred - target) ** 2
        weights = torch.where(target > 0, self.alpha, 1.0)
        L_reg = (weights * mse).sum() / B
        
        # ==================== L_var: Variance Penalty ====================
        # L_var = ||Var(ŷ) - Var(y)||²
        pred_var = pred.var()
        target_var = target.var()
        L_var = (pred_var - target_var) ** 2
        
        # ==================== L_kl: KL Divergence Loss ====================
        # KL(P_ŷ || P_y)
        # Convert predictions and targets to probability distributions
        pred_dist = F.softmax(pred, dim=0)
        target_dist = F.softmax(target, dim=0)
        L_kl = F.kl_div(pred_dist.log(), target_dist, reduction='sum')
        
        # ==================== Total Loss ====================
        # L_total = L_reg + λ1 * L_var + λ2 * KL
        L_total = L_reg + self.lambda1 * L_var + self.lambda2 * L_kl
        
        return L_total


def train_epoch(model, dataloader, optimizer, scheduler, device, scaler, use_amp=True, accumulation_steps=1):
    model.train()
    losses = AverageMeter()
    
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(dataloader):
        # optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        ctr_labels = batch['ctr'].to(device)
        
        with autocast(device_type='cuda', enabled=use_amp):
            ctr_logits, text_emb, img_emb = model(input_ids, attention_mask, pixel_values)
            
            criterion = DistributionAwareCompoundLoss(alpha=10.0, lambda1=0.1, lambda2=0.01)
            loss = criterion(ctr_logits, ctr_labels)
            
            loss = loss / accumulation_steps

        if use_amp: 
            scaler.scale(loss).backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()  
        
        if scheduler is not None and (batch_idx + 1) % accumulation_steps == 0:
            scheduler.step()
        
        # Record original loss (not divided by accumulation_steps)
        losses.update(loss.item() * accumulation_steps, input_ids.size(0))
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {losses.avg:.6f}")
    
    return losses.avg


def evaluate(model, dataloader, device, use_amp=True):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader: 
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            ctr_labels = batch['ctr']
            
            with autocast(device_type='cuda', enabled=use_amp):
                ctr_pred, _, _ = model(input_ids, attention_mask, pixel_values)
            
            all_preds.append(ctr_pred.cpu())
            all_labels.append(ctr_labels)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Ensure tensors are 1D
    all_preds = all_preds.squeeze()
    all_labels = all_labels.squeeze()

    # Compute evaluation metrics
    mse = F.mse_loss(all_preds, all_labels).item()
    mae = F.l1_loss(all_preds, all_labels).item()
    
    # 打印预测分布统计
    pred_mean = all_preds.mean().item()
    pred_std = all_preds.std().item()
    pred_min = all_preds.min().item()
    pred_max = all_preds.max().item()
    
    label_mean = all_labels.mean().item()
    label_std = all_labels.std().item()
    
    print(f"\n  Prediction Distribution:")
    print(f"    Pred: mean={pred_mean:.4f}, std={pred_std:.4f}, min={pred_min:.4f}, max={pred_max:.4f}")
    print(f"    Label: mean={label_mean:.4f}, std={label_std:.4f}")
    
    # If labels have variance, compute correlation
    if label_std > 1e-6:
        correlation = torch.corrcoef(torch.stack([all_preds.flatten(), all_labels.flatten()]))[0, 1].item()
        print(f"    Correlation: {correlation:.4f}")

    # Compute metrics for non-zero samples
    nonzero_mask = all_labels > 0
    if nonzero_mask.sum() > 0:
        nonzero_preds = all_preds[nonzero_mask]
        nonzero_labels = all_labels[nonzero_mask]
        nonzero_mae = F.l1_loss(nonzero_preds, nonzero_labels).item()
        print(f"    Non-zero MAE: {nonzero_mae:.4f} (n={nonzero_mask.sum()})")
    
    return mse, mae
    


def main(args):
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer and image processor
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    image_processor = AutoImageProcessor.from_pretrained(args.image_model)
    
    print(f"\n{'='*50}")
    print("Loading datasets...")
    print(f"{'='*50}")
    
    # Create datasets (will automatically filter invalid images)
    train_dataset = FoodCTRDataset(
        csv_path=args.train_csv,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_seq_len=args.max_seq_len,
        target_size=args.target_image_size,
        resize_mode=args.resize_mode
    )
    
    test_dataset = FoodCTRDataset(
        csv_path=args.test_csv,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_seq_len=args.max_seq_len,
        target_size=args.target_image_size,
        resize_mode=args.resize_mode
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"\n{'='*50}")
    print("Creating model...")
    print(f"{'='*50}")
    
    # Create model
    model = FoodCTRModel(
        bert_model_name=args.bert_model,
        image_model_name=args.image_model,
        hidden_dims=[512, 256],
        fc_hidden_size=[128],
        drop_prob=args.dropout,
        num_transformer_layers=args.num_transformer_layers,
        num_transformer_heads=args.num_transformer_heads,
        freeze_bert=args.freeze_bert,
        freeze_image_backbone=args.freeze_image
    )
    model = model.to(device)
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer - layer-wise learning rates
    bert_param_ids = set(id(p) for p in model.text_encoder.bert.parameters())
    image_param_ids = set(id(p) for p in model.image_encoder.backbone.parameters())
    
    bert_params = [p for p in model.text_encoder.bert.parameters() if p.requires_grad]
    image_params = [p for p in model.image_encoder.backbone.parameters() if p.requires_grad]
    other_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in bert_param_ids and id(p) not in image_param_ids
    ]
    
    print(f"Parameter groups - BERT: {len(bert_params)}, Image: {len(image_params)}, Other: {len(other_params)}")
    
    param_groups = []
    if bert_params: 
        param_groups.append({'params': bert_params, 'lr': args.lr_bert})
    if image_params: 
        param_groups.append({'params':  image_params, 'lr': args.lr_image})
    if other_params: 
        param_groups.append({'params': other_params, 'lr': args.lr})
    
    optimizer = torch.optim.AdamW(
        param_groups, 
        weight_decay=args.weight_decay,
        eps=1e-8  # Numerical stability
    )
    
    # Learning rate scheduler
    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    
    max_lrs = []
    if bert_params:
        max_lrs.append(args.lr_bert)
    if image_params: 
        max_lrs.append(args.lr_image)
    if other_params:
        max_lrs.append(args.lr)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps
    )
    
    # Mixed precision training
    scaler = GradScaler('cuda') if args.use_amp else None
    
    # Training loop
    best_mse = float('inf')
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")
        
        train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler, device, scaler, args.use_amp, args.accumulation_steps
        )
        print(f"Train Loss: {train_loss:.6f}")
        
        mse, mae = evaluate(model, test_dataloader, device, args.use_amp)
        print(f"Test MSE: {mse:.6f}, Test MAE: {mae:.6f}")
        
        if mse < best_mse:
            best_mse = mse
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                'mse': best_mse,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"Saved best model with MSE: {best_mse:.6f}")
        
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict':  model.state_dict(),
                'mse': mse,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    print(f"\nTraining completed!  Best MSE: {best_mse:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Food CTR Model')
    
    # 数据参数
    parser.add_argument('--train-csv', type=str, required=True)
    parser.add_argument('--test-csv', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./output')
    
    # 模型参数
    parser.add_argument('--bert-model', type=str, default='hfl/chinese-bert-wwm-ext')
    parser.add_argument('--image-model', type=str, default='facebook/convnextv2-huge-22k-512')
    parser.add_argument('--max-seq-len', type=int, default=128)
    parser.add_argument('--target-image-size', type=int, default=512)
    parser.add_argument('--resize-mode', type=str, default='padding', choices=['padding', 'stretch'])
    parser.add_argument('--num-transformer-layers', type=int, default=4)
    parser.add_argument('--num-transformer-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--freeze-bert', action='store_true')
    parser.add_argument('--freeze-image', action='store_true')
    parser.add_argument('--accumulation-steps', type=int, default=4)
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr-bert', type=float, default=1e-5)
    parser.add_argument('--lr-image', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)