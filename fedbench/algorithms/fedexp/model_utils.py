import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Any, Optional


def set_seed(seed):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results(metrics, method_name, dataset_name, partition_name, save_dir="./results"):
    """
    Save results to file.
    
    Args:
        metrics: Dictionary with metrics
        method_name: Name of the method
        dataset_name: Name of the dataset
        partition_name: Name of the data partitioning
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics as numpy arrays
    save_path = f"{save_dir}/{method_name}_{dataset_name}_{partition_name}"
    np.savez(
        f"{save_path}_metrics.npz",
        **{k: np.array(v) if isinstance(v, list) else v for k, v in metrics.items()}
    )


def load_results(method_name, dataset_name, partition_name, load_dir="./results"):
    """
    Load results from file.
    
    Args:
        method_name: Name of the method
        dataset_name: Name of the dataset
        partition_name: Name of the data partitioning
        load_dir: Directory to load results from
        
    Returns:
        Dictionary with loaded metrics
    """
    load_path = f"{load_dir}/{method_name}_{dataset_name}_{partition_name}_metrics.npz"
    
    if not os.path.exists(load_path):
        print(f"No results found at {load_path}")
        return {}
    
    loaded = np.load(load_path, allow_pickle=True)
    metrics = {k: loaded[k].tolist() if loaded[k].ndim > 0 else loaded[k].item() for k in loaded.files}
    
    return metrics


def compare_methods(results_dict, dataset_name, partition_name, save_dir="./results/comparisons"):
    """
    Compare results from multiple methods.
    
    Args:
        results_dict: Dictionary mapping method names to their metrics
        dataset_name: Name of the dataset
        partition_name: Name of the data partitioning
        save_dir: Directory to save comparison plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Compare test accuracy
    plt.figure(figsize=(12, 6))
    for method_name, metrics in results_dict.items():
        if 'test_accuracy' in metrics:
            plt.plot(metrics['test_accuracy'], marker='o', label=method_name)
    
    plt.title(f'Test Accuracy Comparison ({dataset_name}, {partition_name} partition)')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_dir}/{dataset_name}_{partition_name}_accuracy_comparison.png")
    plt.close()
    
    # Compare training loss
    plt.figure(figsize=(12, 6))
    for method_name, metrics in results_dict.items():
        if 'train_loss' in metrics:
            plt.plot(metrics['train_loss'], marker='o', label=method_name)
    
    plt.title(f'Training Loss Comparison ({dataset_name}, {partition_name} partition)')
    plt.xlabel('Communication Round')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_dir}/{dataset_name}_{partition_name}_loss_comparison.png")
    plt.close()


def print_client_data_stats(client_data_dict):
    """
    Print statistics about client data distribution.
    
    Args:
        client_data_dict: Dictionary mapping client IDs to their data
    """
    print("\nClient Data Statistics:")
    print("-----------------------")
    
    for client_id, dataloader in client_data_dict.items():
        # Count samples
        num_samples = len(dataloader.dataset)
        
        # Count labels if possible
        labels = []
        if hasattr(dataloader.dataset, 'targets'):
            labels = dataloader.dataset.targets
        elif hasattr(dataloader.dataset, 'labels'):
            labels = dataloader.dataset.labels
        elif hasattr(dataloader.dataset, 'dataset') and hasattr(dataloader.dataset.dataset, 'targets'):
            # For Subset datasets
            indices = dataloader.dataset.indices
            all_targets = dataloader.dataset.dataset.targets
            if isinstance(all_targets, torch.Tensor):
                labels = all_targets[indices].tolist()
            else:
                labels = [all_targets[i] for i in indices]
        
        # Print client stats
        if labels:
            unique_labels = set(labels)
            label_counts = {label: labels.count(label) for label in unique_labels}
            print(f"Client {client_id}: {num_samples} samples, Labels: {label_counts}")
        else:
            print(f"Client {client_id}: {num_samples} samples")