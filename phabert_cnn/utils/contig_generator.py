"""
Contig Generation Module

Generates contigs of varying lengths from complete phage genomes using 
a sliding window approach to simulate real-world metagenomic scenarios.

For each window:
- Length is randomly sampled from a normal distribution
- Consecutive windows overlap by a specified percentage
- Four length categories: A (100-400bp), B (400-800bp), C (800-1200bp), D (1200-1800bp)

Data augmentation: reverse complement sequences for all contigs.
"""

import random
import numpy as np
from typing import List, Tuple, Dict
from Bio.Seq import Seq


def reverse_complement(seq: str) -> str:
    """Generate reverse complement of a DNA sequence."""
    return str(Seq(seq).reverse_complement())


def generate_contigs_from_genome(
    genome_seq: str,
    min_length: int,
    max_length: int,
    overlap_pct: float,
    seed: int = None,
) -> List[str]:
    """
    Generate contigs from a single genome using sliding window approach.
    
    Args:
        genome_seq: Complete genome sequence string
        min_length: Minimum contig length
        max_length: Maximum contig length
        overlap_pct: Overlap percentage between consecutive windows
        seed: Random seed for reproducibility
        
    Returns:
        List of contig sequences
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    genome_len = len(genome_seq)
    contigs = []
    
    # Mean and std for normal distribution of contig lengths
    mean_length = (min_length + max_length) / 2
    std_length = (max_length - min_length) / 6  # ~99.7% within range
    
    pos = 0
    while pos < genome_len:
        # Sample contig length from normal distribution
        contig_len = int(rng.normal(mean_length, std_length))
        contig_len = max(min_length, min(max_length, contig_len))
        
        # Extract contig
        end_pos = min(pos + contig_len, genome_len)
        contig = genome_seq[pos:end_pos]
        
        # Only keep if contig is long enough
        if len(contig) >= min_length:
            contigs.append(contig.upper())
        
        # Move window with overlap
        step = int(contig_len * (1 - overlap_pct))
        step = max(step, 1)  # Ensure at least 1bp step
        pos += step
    
    return contigs


def generate_dataset_contigs(
    genomes: List[Tuple[str, int]],  # (sequence, label)
    group_config: Dict,
    use_reverse_complement: bool = True,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    """
    Generate contigs for an entire dataset of genomes.
    
    Args:
        genomes: List of (genome_sequence, label) tuples
        group_config: Dict with min_length, max_length, overlap_pct
        use_reverse_complement: Whether to add reverse complement augmentation
        seed: Random seed
        
    Returns:
        Tuple of (contig_sequences, labels)
    """
    all_contigs = []
    all_labels = []
    
    for i, (genome_seq, label) in enumerate(genomes):
        contigs = generate_contigs_from_genome(
            genome_seq=genome_seq,
            min_length=group_config['min_length'],
            max_length=group_config['max_length'],
            overlap_pct=group_config['overlap_pct'],
            seed=seed + i,
        )
        
        for contig in contigs:
            all_contigs.append(contig)
            all_labels.append(label)
            
            # Data augmentation: reverse complement
            if use_reverse_complement:
                rc_contig = reverse_complement(contig)
                all_contigs.append(rc_contig)
                all_labels.append(label)
    
    return all_contigs, all_labels


# Group configurations as described in the paper
CONTIG_GROUP_CONFIGS = {
    'A': {'min_length': 100, 'max_length': 400, 'overlap_pct': 0.1},
    'B': {'min_length': 400, 'max_length': 800, 'overlap_pct': 0.2},
    'C': {'min_length': 800, 'max_length': 1200, 'overlap_pct': 0.3},
    'D': {'min_length': 1200, 'max_length': 1800, 'overlap_pct': 0.4},
}
