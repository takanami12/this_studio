"""
PhaBERT-CNN Evaluation Script

Evaluates trained models across all 5 folds for a given contig group 
and aggregates results (mean ± std).

Usage:
    python scripts/evaluate.py --group A
    python scripts/evaluate.py --group A --data_dir data/processed
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.phabert_cnn import PhaBERTCNN
from utils.dataset import PhageContigDataset
from utils.metrics import compute_metrics, print_metrics, aggregate_fold_metrics
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PhaBERT-CNN")
    parser.add_argument("--group", type=str, required=True, choices=['A', 'B', 'C', 'D'])
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints")
    parser.add_argument("--output_dir", type=str, default="results/metrics")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--model_name", type=str, default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_split", type=str, default="val",
                        choices=['val', 'test'], help="Which split to evaluate")
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Run inference and compute metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    use_amp = device.type == 'cuda'

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        bio_features = batch['bio_features'].to(device, non_blocking=True)
        labels = batch['label']

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(input_ids, attention_mask, bio_features)
        probs = torch.softmax(logits.float(), dim=-1)
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    return metrics, np.array(all_probs)


def main():
    args = parse_args()
    
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Evaluating Group {args.group} on {args.eval_split} split")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    all_fold_metrics = []
    
    for fold_idx in range(args.n_folds):
        print(f"\n--- Fold {fold_idx} ---")
        
        # Load checkpoint
        exp_name = f"group_{args.group}_fold_{fold_idx}"
        ckpt_path = Path(args.checkpoint_dir) / exp_name / "best_model.pt"
        
        if not ckpt_path.exists():
            print(f"  WARNING: Checkpoint not found: {ckpt_path}")
            continue
        
        # Load data
        fold_dir = Path(args.data_dir) / f"group_{args.group}" / f"fold_{fold_idx}"
        with open(fold_dir / f"{args.eval_split}.pkl", 'rb') as f:
            eval_data = pickle.load(f)
        
        print(f"  Loaded {len(eval_data['sequences'])} samples")
        
        # Load model + bio-feature normaliser from checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        bio_normalizer = checkpoint.get('bio_normalizer')
        bio_dim = checkpoint.get('bio_feature_dim', 0)

        # Create dataset and dataloader (apply same normaliser used in training)
        dataset = PhageContigDataset(
            eval_data['sequences'], eval_data['labels'],
            tokenizer, args.max_seq_length,
            bio_normalizer=bio_normalizer,
        )
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )

        model = PhaBERTCNN(
            dnabert2_model_name=args.model_name,
            bio_feature_dim=bio_dim,
        )
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict):
            state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model = model.to(device)
        
        # Evaluate
        metrics, probs = evaluate_model(model, dataloader, device)
        all_fold_metrics.append(metrics)
        
        print_metrics(metrics, prefix="  ")
        
        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # ================================================================
    # Aggregate results across folds
    # ================================================================
    if len(all_fold_metrics) > 0:
        print("\n" + "=" * 60)
        print(f"Aggregated Results - Group {args.group} ({args.n_folds}-fold CV)")
        print("=" * 60)
        
        agg = aggregate_fold_metrics(all_fold_metrics)
        
        print(f"  Sensitivity: {agg['sensitivity_mean']:.2f}% ± {agg['sensitivity_std']:.2f}%")
        print(f"  Specificity: {agg['specificity_mean']:.2f}% ± {agg['specificity_std']:.2f}%")
        print(f"  Accuracy:    {agg['accuracy_mean']:.2f}% ± {agg['accuracy_std']:.2f}%")
        
        # Save results
        results = {
            'group': args.group,
            'eval_split': args.eval_split,
            'n_folds': len(all_fold_metrics),
            'fold_metrics': all_fold_metrics,
            'aggregated': agg,
        }
        
        result_path = Path(args.output_dir) / f"group_{args.group}_results.json"
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {result_path}")
    
    # ================================================================
    # Print comparison table (paper results)
    # ================================================================
    print("\n" + "=" * 60)
    print("Reference: Paper Results for PhaBERT-CNN")
    print("=" * 60)
    
    paper_results = {
        'A': {'sn': 82.00, 'sp': 80.15, 'acc': 81.59},
        'B': {'sn': 89.91, 'sp': 80.44, 'acc': 87.91},
        'C': {'sn': 91.12, 'sp': 85.93, 'acc': 90.01},
        'D': {'sn': 88.47, 'sp': 90.95, 'acc': 90.69},
    }
    
    if args.group in paper_results:
        ref = paper_results[args.group]
        print(f"  Group {args.group} (paper):")
        print(f"    Sensitivity: {ref['sn']:.2f}%")
        print(f"    Specificity: {ref['sp']:.2f}%")
        print(f"    Accuracy:    {ref['acc']:.2f}%")


if __name__ == "__main__":
    main()
