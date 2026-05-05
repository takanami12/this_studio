"""
Dataset and DataLoader utilities for PhaBERT-CNN.

Handles:
- Tokenization of DNA sequences using DNABERT-2 tokenizer
- Hand-crafted bio-feature pre-computation (k-mer freq, GC, dinuc OR, GC-skew FFT)
- PyTorch Dataset for phage contig sequences
- DataLoader with proper collation and padding
- Random undersampling for class balance
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from imblearn.under_sampling import RandomUnderSampler
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import numpy as np

from .bio_features import (
    BIO_FEATURE_DIM,
    compute_features_batch,
    fit_normalizer,
    apply_normalizer,
)


class PhageContigDataset(Dataset):
    """
    Pre-tokenizes all sequences and pre-computes bio-features at init time.

    Pre-tokenisation: ~30-40% training time saved.
    Pre-computed bio-features: avoids recomputing the same numpy stats every
    epoch and every DataLoader worker. Memory cost is small
    (N * BIO_FEATURE_DIM * 4 bytes; e.g. 200k * 349 * 4 ~= 280 MB).
    """

    def __init__(
        self,
        sequences: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        bio_normalizer: Optional[Dict[str, np.ndarray]] = None,
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

        print(f"    Computing bio-features ({BIO_FEATURE_DIM}-dim)...",
              end=" ", flush=True)
        raw_features = compute_features_batch(sequences)
        if bio_normalizer is not None:
            raw_features = apply_normalizer(raw_features, bio_normalizer)
        self.bio_features = torch.from_numpy(raw_features)
        print("Done!")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'bio_features': self.bio_features[idx],
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


def _try_load_token_cache(
    cache_dir: Optional[Path],
    n_train: int,
    n_val: int,
    max_length: int,
) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
    """Try loading pre-tokenized {train,val}_tokens.pt from cache_dir.

    Returns (train_cache, val_cache) dicts on success, hoặc None nếu thiếu/
    không khớp (size, max_length). Cache chỉ chứa tokens — bio_features vẫn
    được compute từ sequences ở runtime.
    """
    if cache_dir is None:
        return None
    train_path = cache_dir / "train_tokens.pt"
    val_path = cache_dir / "val_tokens.pt"
    if not (train_path.exists() and val_path.exists()):
        return None

    print(f"  [cache] Found pre-tokenized files in {cache_dir}")
    train_cache = torch.load(train_path, map_location='cpu', weights_only=False)
    val_cache = torch.load(val_path, map_location='cpu', weights_only=False)

    # Validate token tensor shape khớp với sequences hiện tại
    for name, cache, expected_n in [
        ('train', train_cache, n_train),
        ('val', val_cache, n_val),
    ]:
        n_cached = cache['input_ids'].shape[0]
        cached_max_len = cache['input_ids'].shape[1]
        if n_cached != expected_n:
            print(f"  [cache] {name}: số sample không khớp "
                  f"(cache={n_cached}, sequences={expected_n}). Bỏ qua cache.")
            return None
        if cached_max_len != max_length:
            print(f"  [cache] {name}: max_length không khớp "
                  f"(cache={cached_max_len}, requested={max_length}). Bỏ qua cache.")
            return None

    print(f"  [cache] Loaded tokens: train={train_cache['input_ids'].shape}, "
          f"val={val_cache['input_ids'].shape}")
    return train_cache, val_cache


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
    cache_dir: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, np.ndarray]]:
    """
    Create training and validation DataLoaders.

    Bio-feature normaliser is fitted on the (post-undersampling) training set
    and applied to both splits, then returned so the caller can persist it
    in the checkpoint for evaluation.

    Nếu `cache_dir/{train,val}_tokens.pt` tồn tại và khớp size/max_length,
    tokens sẽ được load từ cache thay vì tokenize lại; bio_features vẫn
    luôn compute từ sequences (cache không lưu bio_features).

    Returns:
        (train_loader, val_loader, bio_normalizer)
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

    # Try cache. Nếu undersampling đã chạy, kích thước khớp cache cũ chỉ khi
    # cache cũng được tạo bằng cùng undersampling; _try_load_token_cache sẽ
    # tự động phát hiện và bỏ qua nếu lệch.
    cached = _try_load_token_cache(cache_dir, len(train_seqs), len(val_seqs), max_length)

    if cached is not None:
        train_cache, val_cache = cached
        bio_in_cache = ('bio_features' in train_cache and 'bio_features' in val_cache)

        if bio_in_cache:
            print("  [cache] Reusing pre-computed bio-features from cache")
            train_bio = train_cache['bio_features']
            val_bio = val_cache['bio_features']
            bio_normalizer = train_cache.get('bio_normalizer')
            if bio_normalizer is None:
                # Cache cũ chỉ có features mà thiếu normalizer → fit lại từ
                # train sequences để evaluate.py có thể tái sử dụng stats.
                print("  [cache] bio_normalizer missing → fitting fresh from train")
                bio_normalizer = fit_normalizer(compute_features_batch(train_seqs))
        else:
            print("  [cache] Tokens cached but bio-features missing — "
                  "computing fresh from sequences")
            train_raw = compute_features_batch(train_seqs)
            bio_normalizer = fit_normalizer(train_raw)
            train_bio = torch.from_numpy(apply_normalizer(train_raw, bio_normalizer))
            print(f"    Computing val bio-features ({BIO_FEATURE_DIM}-dim)...",
                  end=" ", flush=True)
            val_raw = compute_features_batch(val_seqs)
            val_bio = torch.from_numpy(apply_normalizer(val_raw, bio_normalizer))
            print("Done!")

        train_dataset = _CachedPhageContigDataset(
            input_ids=train_cache['input_ids'],
            attention_masks=train_cache['attention_mask'],
            labels=torch.tensor(train_labels, dtype=torch.long),
            bio_features=train_bio,
            max_length=max_length,
        )
        val_dataset = _CachedPhageContigDataset(
            input_ids=val_cache['input_ids'],
            attention_masks=val_cache['attention_mask'],
            labels=torch.tensor(val_labels, dtype=torch.long),
            bio_features=val_bio,
            max_length=max_length,
        )
    else:
        # Fit normaliser on raw training features
        print("  Fitting bio-feature normaliser on training split...")
        train_raw_features = compute_features_batch(train_seqs)
        bio_normalizer = fit_normalizer(train_raw_features)
        train_norm = apply_normalizer(train_raw_features, bio_normalizer)

        train_dataset = _PrebuiltPhageContigDataset(
            sequences=train_seqs,
            labels=train_labels,
            tokenizer=tokenizer,
            max_length=max_length,
            normalized_features=train_norm,
        )
        val_dataset = PhageContigDataset(
            sequences=val_seqs,
            labels=val_labels,
            tokenizer=tokenizer,
            max_length=max_length,
            bio_normalizer=bio_normalizer,
        )

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

    return train_loader, val_loader, bio_normalizer


class _PrebuiltPhageContigDataset(PhageContigDataset):
    """Internal variant that accepts already-normalised features.

    Used by create_dataloaders to avoid recomputing bio-features twice on the
    training split (once for fitting the normaliser, once inside __init__).
    """

    def __init__(
        self,
        sequences: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int,
        normalized_features: np.ndarray,
    ):
        # Skip the parent __init__'s feature computation by initialising the
        # tokenisation manually, then assigning the cached feature tensor.
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.max_length = max_length

        print(f"    Pre-tokenizing {len(sequences)} sequences...", end=" ", flush=True)
        BATCH = 10000
        all_input_ids = []
        all_attention_masks = []
        for i in range(0, len(sequences), BATCH):
            batch_seqs = sequences[i:i + BATCH]
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

        self.bio_features = torch.from_numpy(normalized_features)


class _CachedPhageContigDataset(Dataset):
    """Dataset xây từ tokens đã pre-tokenized (cache .pt) + bio_features
    đã chuẩn hoá sẵn ở caller. Không gọi tokenizer.
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        labels: torch.Tensor,
        bio_features: torch.Tensor,
        max_length: int,
    ):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.bio_features = bio_features
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'bio_features': self.bio_features[idx],
            'label': self.labels[idx],
        }
