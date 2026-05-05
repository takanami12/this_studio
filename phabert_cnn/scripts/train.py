"""
PhaBERT-CNN Training Script

Implements the two-phase progressive fine-tuning strategy:

Phase 1 (Warm-up):
    - Freeze DNABERT-2 backbone
    - Train CNN + attention + classifier for 1 epoch
    - AdamW optimizer, lr=2e-3, weight_decay=1e-4
    - OneCycleLR with 30% warm-up

Phase 2 (Full Fine-tuning):
    - Unfreeze all layers
    - Discriminative learning rates:
        - DNABERT-2: lr=1e-5, wd=1e-5
        - Task layers: lr=1e-4, wd=1e-4
    - OneCycleLR with 10% warm-up
    - Early stopping: patience=3, monitor=val_accuracy
    - Up to 10 epochs

Usage:
    python scripts/train.py --group A --fold 0
    python scripts/train.py --group B --fold 0 --batch_size 16
"""

import os
import sys
import time
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phabert_cnn import PhaBERTCNN
from utils.dataset import create_dataloaders, apply_undersampling
from utils.metrics import compute_metrics, print_metrics
from utils.bio_features import BIO_FEATURE_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="Train PhaBERT-CNN")
    parser.add_argument("--group", type=str, required=True, choices=['A', 'B', 'C', 'D'],
                        help="Contig length group")
    parser.add_argument("--fold", type=int, required=True, choices=range(5),
                        help="Cross-validation fold (0-4)")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Processed data directory")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not specified)")
    parser.add_argument("--seed", type=int, default=42)
    
    # Model args
    parser.add_argument("--model_name", type=str, default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--use_bio_features", dest="use_bio_features",
                        action="store_true", default=True,
                        help="Enable hand-crafted bio-feature MLP branch (default: enabled).")
    parser.add_argument("--no_bio_features", dest="use_bio_features",
                        action="store_false",
                        help="Disable bio-feature branch (paper-baseline ablation).")
    parser.add_argument("--compile", dest="compile",
                        action="store_true", default=True,
                        help="Bật torch.compile cho Phase 2 (mặc định: bật).")
    parser.add_argument("--no_compile", dest="compile",
                        action="store_false",
                        help="Tắt torch.compile (tránh dynamo graph-break warnings "
                             "từ DNABERT-2 bert_padding.unpad_input).")

    # Phase 1 args
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--warmup_lr", type=float, default=2e-3)
    parser.add_argument("--warmup_wd", type=float, default=1e-4)
    
    # Phase 2 args
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--backbone_wd", type=float, default=1e-5)
    parser.add_argument("--task_lr", type=float, default=1e-4)
    parser.add_argument("--task_wd", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)

    # Class imbalance handling
    parser.add_argument("--class_balance", type=str, default="weight",
                        choices=["weight", "undersample", "none"],
                        help="Strategy for class imbalance: 'weight' (default, "
                             "no data loss), 'undersample' (random under-sample "
                             "majority), 'none'.")
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # Input luôn được pad về max_length cố định → benchmark an toàn và nhanh hơn
    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler=None):
    """Train for one epoch with AMP support."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    use_amp = scaler is not None

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        bio_features = batch['bio_features'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(input_ids, attention_mask, bio_features)
            loss = criterion(logits, labels)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_amp=False):
    """Evaluate with AMP support."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        bio_features = batch['bio_features'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(input_ids, attention_mask, bio_features)
            loss = criterion(logits, labels)
        
        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    
    return avg_loss, metrics

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Auto-detect device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # ================================================================
    # Setup directories
    # ================================================================
    exp_name = f"group_{args.group}_fold_{args.fold}"
    checkpoint_dir = Path(args.output_dir) / "checkpoints" / exp_name
    log_dir = Path(args.output_dir) / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Load data
    # ================================================================
    print(f"\nLoading data: Group {args.group}, Fold {args.fold}")
    fold_dir = Path(args.data_dir) / f"group_{args.group}" / f"fold_{args.fold}"
    
    with open(fold_dir / "train.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open(fold_dir / "val.pkl", 'rb') as f:
        val_data = pickle.load(f)
    
    print(f"  Train: {len(train_data['sequences'])} samples")
    print(f"  Val:   {len(val_data['sequences'])} samples")
    
    # ================================================================
    # Initialize tokenizer and dataloaders
    # ================================================================
    print(f"\nLoading DNABERT-2 tokenizer from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    train_loader, val_loader, bio_normalizer = create_dataloaders(
        train_seqs=train_data['sequences'],
        train_labels=train_data['labels'],
        val_seqs=val_data['sequences'],
        val_labels=val_data['labels'],
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
        num_workers=args.num_workers,
        use_undersampling=(args.class_balance == "undersample"),
        random_state=args.seed,
        cache_dir=fold_dir,
    )

    # ================================================================
    # Initialize model
    # ================================================================
    print("\nInitializing PhaBERT-CNN model...")
    bio_dim = BIO_FEATURE_DIM if args.use_bio_features else 0
    print(f"  use_bio_features = {args.use_bio_features} (bio_dim={bio_dim})")
    model = PhaBERTCNN(
        dnabert2_model_name=args.model_name,
        embedding_dim=768,
        cnn_kernel_sizes=[3, 5, 7],
        attention_hidden_dim=64,
        attention_dropout=0.1,
        bio_feature_dim=bio_dim,
        classifier_hidden_dim=256,
        classifier_dropout=0.1,
        num_classes=2,
    )
    model = model.to(device)

    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    if args.class_balance == "weight":
        labels_arr = np.asarray(train_data['labels'])
        n_total = labels_arr.size
        n_pos = int((labels_arr == 1).sum())
        n_neg = int((labels_arr == 0).sum())
        # sklearn 'balanced' formula: n_total / (n_classes * n_per_class)
        weight = torch.tensor(
            [n_total / (2 * n_neg), n_total / (2 * n_pos)],
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=weight)
        print(f"  Class weights (weight strategy): "
              f"temperate={weight[0].item():.3f}, virulent={weight[1].item():.3f}")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"  Class balance strategy: {args.class_balance} (uniform CE loss)")

    # AMP scaler for mixed precision training
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("  Mixed precision (AMP) enabled")
    
    # Training log
    training_log = {
        'args': vars(args),
        'phases': [],
    }
    
    # ================================================================
    # Phase 1: Warm-up (Freeze DNABERT-2, train task layers)
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 1: Warm-up (Freeze DNABERT-2)")
    print("=" * 60)
    
    model.freeze_backbone()
    trainable_params_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters (Phase 1): {trainable_params_p1:,}")
    
    # DEBUG: Skip warmup if checkpoint exists from previous run
    warmup_skip_path = checkpoint_dir / 'after_warmup.pt'
    if warmup_skip_path.exists():
        print("  [DEBUG] Loading saved warmup checkpoint, skipping Phase 1...")
        ckpt = torch.load(warmup_skip_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        # Phase 1 optimizer: only task parameters
        optimizer_p1 = AdamW(
            model.get_task_params(),
            lr=args.warmup_lr,
            weight_decay=args.warmup_wd,
            betas=(0.9, 0.999),
        )
        
        # Phase 1 scheduler: OneCycleLR with 30% warm-up
        total_steps_p1 = args.warmup_epochs * len(train_loader)
        scheduler_p1 = OneCycleLR(
            optimizer_p1,
            max_lr=args.warmup_lr,
            total_steps=total_steps_p1,
            pct_start=0.3,
            div_factor=5,
            final_div_factor=10,
        )
        
        for epoch in range(args.warmup_epochs):
            print(f"\n--- Warm-up Epoch {epoch+1}/{args.warmup_epochs} ---")
            
            train_loss, train_metrics = train_one_epoch(
                model, train_loader, optimizer_p1, scheduler_p1, criterion, device, scaler
            )
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
            
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_metrics['accuracy']:.2f}%")
            print_metrics(val_metrics, prefix="  Val ")
            
            training_log['phases'].append({
                'phase': 'warmup',
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_metrics': train_metrics,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            })
        
        # Save after warmup for fast Phase 2 debugging next time
        torch.save({'model_state_dict': model.state_dict()}, warmup_skip_path)
        print(f"  [DEBUG] Warmup checkpoint saved to: {warmup_skip_path}")

    # Compile AFTER Phase 1 (cả cache hit lẫn cache miss) để Phase 2 luôn được compile
    if args.compile and hasattr(torch, 'compile'):
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # ================================================================
    # Phase 2: Full Fine-tuning (Discriminative LR)
    # ================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Full Fine-tuning (Discriminative LR)")
    print("=" * 60)
    
    model.unfreeze_backbone()
    trainable_params_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters (Phase 2): {trainable_params_p2:,}")
    
    # Phase 2 optimizer: discriminative learning rates
    optimizer_p2 = AdamW([
        {
            'params': list(model.get_backbone_params()),
            'lr': args.backbone_lr,
            'weight_decay': args.backbone_wd,
        },
        {
            'params': list(model.get_task_params()),
            'lr': args.task_lr,
            'weight_decay': args.task_wd,
        },
    ], betas=(0.9, 0.999), eps=1e-6)
    
    # Phase 2 scheduler: OneCycleLR with 10% warm-up
    total_steps_p2 = args.finetune_epochs * len(train_loader)
    scheduler_p2 = OneCycleLR(
        optimizer_p2,
        max_lr=[args.backbone_lr, args.task_lr],
        total_steps=total_steps_p2,
        pct_start=0.1,
        div_factor=5,
        final_div_factor=50,
    )
    
    # Early stopping
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(args.finetune_epochs):
        print(f"\n--- Fine-tune Epoch {epoch+1}/{args.finetune_epochs} ---")
        
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer_p2, scheduler_p2, criterion, device, scaler
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_metrics['accuracy']:.2f}%")
        print_metrics(val_metrics, prefix="  Val ")
        
        training_log['phases'].append({
            'phase': 'finetune',
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_metrics': train_metrics,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
        })
        
        # Early stopping check
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'val_loss': val_loss,
                'bio_normalizer': bio_normalizer,
                'bio_feature_dim': bio_dim,
            }, checkpoint_dir / 'best_model.pt')
            print(f"  >> New best model saved (val_acc={best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  >> No improvement ({patience_counter}/{args.patience})")
            
            if patience_counter >= args.patience:
                print(f"\n  Early stopping triggered at epoch {epoch+1}")
                break
    
    # ================================================================
    # Save training log
    # ================================================================
    training_log['best_epoch'] = best_epoch
    training_log['best_val_acc'] = best_val_acc
    
    log_path = log_dir / f"{exp_name}_log.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"  Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"  Best model saved to: {checkpoint_dir / 'best_model.pt'}")
    print(f"  Training log saved to: {log_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
