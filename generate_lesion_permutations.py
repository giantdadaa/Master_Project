#!/usr/bin/env python3
"""
Generate and save fixed lesion permutations for reproducible experiments.
These permutations can be loaded and reused across different RNN tests.
"""

import torch
import numpy as np
from pathlib import Path
from utils import load_trained_model


def generate_lesion_permutations(model_name='test_0', n_permutations=10, save_dir=None):
    """
    Generate fixed lesion permutations for reproducible experiments.

    Args:
        model_name: Name of the trained model (to get number of neurons)
        n_permutations: Number of permutation vectors to generate
        save_dir: Directory to save permutations (if None, saves to results/model_name)

    Returns:
        permutations: List of torch.LongTensor permutations
        n_neurons: Number of neurons in the model
    """
    # Load model to get number of neurons
    model, params = load_trained_model(model_name)
    n_neurons = params['model']['n']

    print(f"Generating {n_permutations} lesion permutations for {n_neurons} neurons")

    permutations = []
    for i in range(n_permutations):
        # Use a fixed seed for reproducibility, or just generate randomly
        # Setting a seed ensures same permutations every time this script runs
        torch.manual_seed(i)  # Different seed for each permutation
        perm = torch.randperm(n_neurons)
        permutations.append(perm)
        print(f"  Permutation {i}: first 10 indices = {perm[:10].tolist()}")

    # Save permutations
    if save_dir is None:
        save_dir = Path.cwd() / 'results' / model_name
    else:
        save_dir = Path(save_dir)

    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / 'lesion_permutations.pt'

    # Save as dictionary with metadata
    save_data = {
        'permutations': permutations,
        'n_permutations': n_permutations,
        'n_neurons': n_neurons,
        'model_name': model_name,
        'description': 'Fixed lesion permutations for reproducible experiments'
    }
    torch.save(save_data, save_path)

    print(f"\nPermutations saved to: {save_path}")
    print(f"Load with: permutations = torch.load('{save_path}')['permutations']")

    return permutations, n_neurons


def load_lesion_permutations(model_name='test_0', perm_path=None):
    """
    Load pre-generated lesion permutations.

    Args:
        model_name: Name of the trained model (used to find default path)
        perm_path: Path to permutation file (if None, uses default)

    Returns:
        permutations: List of torch.LongTensor permutations
        n_neurons: Number of neurons
        n_permutations: Number of permutations
    """
    if perm_path is None:
        perm_path = Path.cwd() / 'results' / model_name / 'lesion_permutations.pt'
    else:
        perm_path = Path(perm_path)

    if not perm_path.exists():
        raise FileNotFoundError(f"Permutation file not found: {perm_path}")

    save_data = torch.load(perm_path, map_location='cpu', weights_only=False)

    permutations = save_data['permutations']
    n_neurons = save_data['n_neurons']
    n_permutations = save_data['n_permutations']

    print(f"Loaded {n_permutations} lesion permutations from {perm_path}")
    print(f"  Model: {save_data['model_name']}, Neurons: {n_neurons}")

    return permutations, n_neurons, n_permutations


def main(name, n_permutations):
    """Generate permutations when script is run directly."""

    generate_lesion_permutations(
        model_name=name,
        n_permutations=n_permutations
    )


if __name__ == "__main__":
    main(name='test_0', n_permutations=10)