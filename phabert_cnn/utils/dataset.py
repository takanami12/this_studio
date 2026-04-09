"""
Dataset and DataLoader utilities for PhaBERT-CNN.

Handles:
- Tokenization of DNA sequences using DNABERT-2 tokenizer
- PyTorch Dataset for phage contig sequences
- DataLoader with proper collation and padding
- Random undersampling for class balance
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from imblearn.under_sampling import RandomUnderSampler
from typing import List, Tuple, Optional
import numpy as np


class PhageContigDataset(Dataset):
    """
    Pre-tokenizes all sequences at init time instead of per-sample.
    Saves ~30-40% training time by avoiding repeated tokenization.
    """
    
    def __init__(
        self,
        sequences: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.max_length = max_length
        
        # Pre-tokenize ALL sequences at once (batch tokenization is much faster)
        print(f"    Pre-tokenizing {len(sequences)} sequences...", end=" ", flush=True)
        
        BATCH = 10000
        all_input_ids = []
        all_attention_masks = []
        
        for i in range(0, len(sequences), BATCH):
            batch_seqs = sequences[i:i+BATCH]
            encoding = tokenizer(
                batch_seqs,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            all_input_ids.append(encoding['input_ids'])
            all_attention_masks.append(encoding['attention_mask'])
        
        self.input_ids = torch.cat(all_input_ids, dim=0)
        self.attention_masks = torch.cat(all_attention_masks, dim=0)
        print("Done!")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'label': self.labels[idx],
        }


def apply_undersampling(
    sequences: List[str],
    labels: List[int],
    random_state: int = 42,
) -> Tuple[List[str], List[int]]:
    """
    Apply random undersampling to balance class distribution.
    
    The paper uses random undersampling on the training set to address
    the class imbalance (virulent:temperate ≈ 2:1).
    
    Args:
        sequences: List of DNA sequences
        labels: List of labels (0=temperate, 1=virulent)
        random_state: Random seed
        
    Returns:
        Balanced (sequences, labels) tuple
    """
    # RandomUnderSampler needs 2D array
    X = np.arange(len(sequences)).reshape(-1, 1)
    y = np.array(labels)
    
    rus = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    
    indices = X_resampled.flatten()
    balanced_sequences = [sequences[i] for i in indices]
    balanced_labels = y_resampled.tolist()
    
    return balanced_sequences, balanced_labels


def create_dataloaders(
    train_seqs: List[str],
    train_labels: List[int],
    val_seqs: List[str],
    val_labels: List[int],
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 4,
    use_undersampling: bool = True,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.
    
    Args:
        train_seqs, train_labels: Training data
        val_seqs, val_labels: Validation data
        tokenizer: DNABERT-2 tokenizer
        batch_size: Batch size
        max_length: Max sequence length in tokens
        num_workers: Number of data loading workers
        use_undersampling: Whether to apply random undersampling on train set
        random_state: Random seed
        
    Returns:
        (train_loader, val_loader) tuple
    """
    # Apply undersampling on training set only
    if use_undersampling:
        train_seqs, train_labels = apply_undersampling(
            train_seqs, train_labels, random_state
        )
        print(f"  After undersampling: {len(train_seqs)} training samples")
        unique, counts = np.unique(train_labels, return_counts=True)
        for u, c in zip(unique, counts):
            label_name = "temperate" if u == 0 else "virulent"
            print(f"    {label_name}: {c}")
    
    # Create datasets
    train_dataset = PhageContigDataset(train_seqs, train_labels, tokenizer, max_length)
    val_dataset = PhageContigDataset(val_seqs, val_labels, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    
    return train_loader, val_loader
