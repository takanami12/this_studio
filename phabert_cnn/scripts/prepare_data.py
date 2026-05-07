"""
Data Preparation Pipeline

This script handles the complete data preparation pipeline:
1. Load raw phage genome FASTA files placed manually in --data_dir
2. Create stratified 5-fold cross-validation splits
3. Generate contigs for each length group (A, B, C, D)
4. Apply reverse complement augmentation
5. Save processed data for training

Usage:
    python scripts/prepare_data.py [--data_dir data/raw] [--output_dir data/processed]

Note:
    This script does NOT download data. Place your FASTA files in --data_dir
    (default: data/raw/) before running. File names must contain the keyword
    "virulent" or "temperate" (case-insensitive). Supported extensions:
    .fasta, .fa, .fna, .fsa.
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import StratifiedKFold
from Bio import SeqIO

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.contig_generator import generate_dataset_contigs, CONTIG_GROUP_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare PhaBERT-CNN dataset")
    parser.add_argument("--data_dir", type=str, default="data/raw",
                        help="Directory containing raw FASTA files")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Directory for processed data")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train/val split ratio within each fold")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def _find_fasta_files(search_dir: str, keyword: str) -> List[str]:
    """Recursively find FASTA files whose names contain `keyword`."""
    extensions = ['.fasta', '.fa', '.fna', '.fsa']
    found = []
    search_path = Path(search_dir)

    if not search_path.exists():
        return found

    for ext in extensions:
        for f in search_path.rglob(f"*{ext}"):
            if keyword.lower() in f.name.lower():
                found.append(str(f))

    found.sort()
    return found


def load_fasta_sequences(fasta_path: str) -> List[Tuple[str, str]]:
    """Load (header, sequence) pairs from a FASTA file."""
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append((record.id, str(record.seq)))
    return sequences


def prepare_genome_dataset(data_dir: str) -> List[Tuple[str, int]]:
    """
    Scan data_dir for FASTA files containing "virulent" or "temperate" in
    their filenames (case-insensitive) and load all sequences.

    Files with "delet" in the name are skipped (e.g. Virulent_delet.fasta).
    """
    genomes = []

    vir_files = [f for f in _find_fasta_files(data_dir, "virulent")
                 if 'delet' not in Path(f).name.lower()]
    temp_files = [f for f in _find_fasta_files(data_dir, "temperate")
                  if 'delet' not in Path(f).name.lower()]

    for fpath in vir_files:
        print(f"Loading virulent phages from: {fpath}")
        seqs = load_fasta_sequences(fpath)
        for _, seq in seqs:
            genomes.append((seq, 1))  # virulent = 1
        print(f"  -> {len(seqs)} genomes loaded")

    for fpath in temp_files:
        print(f"Loading temperate phages from: {fpath}")
        seqs = load_fasta_sequences(fpath)
        for _, seq in seqs:
            genomes.append((seq, 0))  # temperate = 0
        print(f"  -> {len(seqs)} genomes loaded")

    num_vir = sum(1 for _, l in genomes if l == 1)
    num_temp = sum(1 for _, l in genomes if l == 0)
    print(f"\nTotal loaded: {len(genomes)} genomes "
          f"({num_vir} virulent, {num_temp} temperate)")

    return genomes


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: data directory not found: {args.data_dir}")
        print("Please create it and place your FASTA files inside, e.g.:")
        print(f"  {args.data_dir}/virulent.fasta   (virulent phage genomes)")
        print(f"  {args.data_dir}/temperate.fasta  (temperate phage genomes)")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ================================================================
    # Step 1: Load raw genome data
    # ================================================================
    print("=" * 60)
    print("Step 1: Loading phage genome data")
    print("=" * 60)

    genomes = prepare_genome_dataset(args.data_dir)

    if len(genomes) == 0:
        print(f"\nNo FASTA files found in {args.data_dir}.")
        print("Expected file names containing 'virulent' or 'temperate' "
              "(case-insensitive), e.g.:")
        print(f"  {args.data_dir}/virulent.fasta")
        print(f"  {args.data_dir}/temperate.fasta")
        print("Supported extensions: .fasta, .fa, .fna, .fsa.")
        sys.exit(1)

    # ================================================================
    # Step 2: Create stratified 5-fold cross-validation splits
    # ================================================================
    print("\n" + "=" * 60)
    print(f"Step 2: Creating {args.n_folds}-fold stratified cross-validation splits")
    print("=" * 60)

    sequences = [g[0] for g in genomes]
    labels = [g[1] for g in genomes]

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    folds = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(sequences, labels)):
        train_val_seqs = [sequences[i] for i in train_val_idx]
        train_val_labels = [labels[i] for i in train_val_idx]

        from sklearn.model_selection import train_test_split
        train_seqs, val_seqs, train_labels, val_labels = train_test_split(
            train_val_seqs, train_val_labels,
            train_size=args.train_ratio,
            stratify=train_val_labels,
            random_state=args.seed + fold_idx,
        )

        test_seqs = [sequences[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        fold_data = {
            'train': (train_seqs, train_labels),
            'val': (val_seqs, val_labels),
            'test': (test_seqs, test_labels),
        }
        folds.append(fold_data)

        n_train_vir = sum(1 for l in train_labels if l == 1)
        n_train_temp = sum(1 for l in train_labels if l == 0)
        print(f"  Fold {fold_idx}: train={len(train_seqs)} "
              f"(vir={n_train_vir}, temp={n_train_temp}), "
              f"val={len(val_seqs)}, test={len(test_seqs)}")

    # ================================================================
    # Step 3: Generate contigs for each group and fold
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 3: Generating contigs for each group and fold")
    print("=" * 60)

    for group_name, group_config in CONTIG_GROUP_CONFIGS.items():
        print(f"\n--- Group {group_name}: {group_config['min_length']}-{group_config['max_length']}bp ---")

        group_dir = Path(args.output_dir) / f"group_{group_name}"
        group_dir.mkdir(parents=True, exist_ok=True)

        for fold_idx, fold_data in enumerate(folds):
            print(f"  Fold {fold_idx}:")
            fold_dir = group_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            for split_name in ['train', 'val', 'test']:
                genome_seqs, genome_labels = fold_data[split_name]
                genome_pairs = list(zip(genome_seqs, genome_labels))

                contig_seqs, contig_labels = generate_dataset_contigs(
                    genomes=genome_pairs,
                    group_config=group_config,
                    use_reverse_complement=True,
                    seed=args.seed + fold_idx * 100 + ord(group_name),
                )

                data = {
                    'sequences': contig_seqs,
                    'labels': contig_labels,
                }

                save_path = fold_dir / f"{split_name}.pkl"
                with open(save_path, 'wb') as f:
                    pickle.dump(data, f)

                n_vir = sum(1 for l in contig_labels if l == 1)
                n_temp = sum(1 for l in contig_labels if l == 0)
                print(f"    {split_name}: {len(contig_seqs)} contigs "
                      f"(vir={n_vir}, temp={n_temp})")

    # ================================================================
    # Save metadata
    # ================================================================
    metadata = {
        'n_genomes': len(genomes),
        'n_virulent': sum(1 for _, l in genomes if l == 1),
        'n_temperate': sum(1 for _, l in genomes if l == 0),
        'n_folds': args.n_folds,
        'groups': list(CONTIG_GROUP_CONFIGS.keys()),
        'seed': args.seed,
    }

    with open(Path(args.output_dir) / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Processed data saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
