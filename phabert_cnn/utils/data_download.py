"""
Data Download Utility

Downloads raw phage genome data from DeePhage and DeepPL sources.
The combined dataset includes 2,241 complete phage genomes:
- 707 temperate phages (label 0)
- 1,534 virulent phages (label 1)

Data sources:
- DeePhage: https://github.com/shufangwu/DeePhage
- DeepPL: Associated with Zhang et al. (2024), PLOS Computational Biology
"""

import os
import subprocess
from pathlib import Path
from typing import List, Tuple
from Bio import SeqIO


def download_deephage_data(output_dir: str) -> str:
    """
    Download DeePhage dataset from GitHub.
    
    Returns path to downloaded directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    deephage_dir = output_dir / "deephage"
    
    if not deephage_dir.exists():
        print("Downloading DeePhage dataset...")
        subprocess.run([
            "git", "clone",
            "https://github.com/shufangwu/DeePhage.git",
            str(deephage_dir),
        ], check=True)
        print("DeePhage dataset downloaded.")
    else:
        print("DeePhage dataset already exists, skipping download.")
    
    return str(deephage_dir)


def download_deeppl_data(output_dir: str) -> str:
    """
    Download DeepPL supplementary data.
    
    The DeepPL paper provides additional labeled sequences.
    Users may need to download manually from the paper's supplementary materials.
    
    Returns path to downloaded directory.
    """
    output_dir = Path(output_dir)
    deeppl_dir = output_dir / "deeppl"
    deeppl_dir.mkdir(parents=True, exist_ok=True)
    
    # NOTE: DeepPL data may need manual download from:
    # https://github.com/li-bw18/DeepPL or the paper's supplementary materials
    
    if not (deeppl_dir / "data").exists():
        print("="*60)
        print("NOTE: DeepPL additional data may need manual download.")
        print("Please download from the DeepPL paper's supplementary")
        print(f"materials and place in: {deeppl_dir}")
        print("="*60)
    
    return str(deeppl_dir)


def load_fasta_sequences(fasta_path: str) -> List[Tuple[str, str]]:
    """
    Load sequences from a FASTA file.
    
    Returns list of (header, sequence) tuples.
    """
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append((record.id, str(record.seq)))
    return sequences

def _find_fasta_files(search_dir: str, keyword: str) -> List[str]:
    """
    Recursively find all FASTA files containing a keyword in filename.
    
    Supports extensions: .fasta, .fa, .fna, .fsa
    """
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


def prepare_genome_dataset(
    data_dir: str,
    virulent_fasta: str = None,
    temperate_fasta: str = None,
) -> List[Tuple[str, int]]:
    """
    Prepare combined genome dataset.
    
    Auto-detects FASTA files by scanning for filenames containing 
    'virulent' or 'temperate' (case-insensitive) in data_dir and 
    all subdirectories.
    
    Files with 'delet' in the name are skipped (e.g., Virulent_delet.fasta).
    """
    genomes = []
    
    # --- Collect virulent FASTA files ---
    if virulent_fasta and os.path.exists(virulent_fasta):
        vir_files = [virulent_fasta]
    else:
        vir_files = _find_fasta_files(data_dir, "virulent")
        vir_files = [f for f in vir_files if 'delet' not in Path(f).name.lower()]
    
    # --- Collect temperate FASTA files ---
    if temperate_fasta and os.path.exists(temperate_fasta):
        temp_files = [temperate_fasta]
    else:
        temp_files = _find_fasta_files(data_dir, "temperate")
        temp_files = [f for f in temp_files if 'delet' not in Path(f).name.lower()]
    
    # --- Load virulent genomes ---
    for fpath in vir_files:
        print(f"Loading virulent phages from: {fpath}")
        seqs = load_fasta_sequences(fpath)
        for header, seq in seqs:
            genomes.append((seq, 1))  # virulent = 1
        print(f"  -> {len(seqs)} genomes loaded")
    
    # --- Load temperate genomes ---
    for fpath in temp_files:
        print(f"Loading temperate phages from: {fpath}")
        seqs = load_fasta_sequences(fpath)
        for header, seq in seqs:
            genomes.append((seq, 0))  # temperate = 0
        print(f"  -> {len(seqs)} genomes loaded")
    
    # --- Summary ---
    if not genomes:
        print("\nWARNING: No FASTA files found automatically.")
        print(f"Searched recursively in: {data_dir}")
        print("Looking for files with 'virulent' or 'temperate' in the filename.")
        
    num_vir = sum(1 for _, l in genomes if l == 1)
    num_temp = sum(1 for _, l in genomes if l == 0)
    print(f"\nTotal loaded: {len(genomes)} genomes ({num_vir} virulent, {num_temp} temperate)")
    
    return genomes
