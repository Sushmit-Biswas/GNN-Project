"""
Utility functions for the GNN project.
Provides seed setting, device detection, homophily computation, and timing.
"""

import os
import time
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for full reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(False)


def get_device():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU (no CUDA GPU detected)")
    return device


def compute_homophily(edge_index, labels):
    """
    Compute edge homophily ratio: fraction of edges connecting same-class nodes.
    
    Homophily h ∈ [0, 1]:
      - h ≈ 1.0 → strong homophily (neighbors share labels)
      - h ≈ 0.0 → strong heterophily (neighbors differ)
    
    Args:
        edge_index: [2, E] tensor of edge indices
        labels: [N] tensor of node labels
    
    Returns:
        float: homophily ratio
    """
    src, dst = edge_index[0], edge_index[1]
    same_label = (labels[src] == labels[dst]).float()
    return same_label.mean().item()


def compute_node_homophily(edge_index, labels, num_nodes):
    """
    Compute per-node homophily: for each node, fraction of neighbors with same label.
    
    Args:
        edge_index: [2, E] tensor of edge indices
        labels: [N] tensor of node labels
        num_nodes: total number of nodes
    
    Returns:
        np.ndarray: per-node homophily ratios of shape [N]
    """
    node_homophily = np.zeros(num_nodes)
    node_degree = np.zeros(num_nodes)
    
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    for i in range(len(src)):
        s, d = src[i], dst[i]
        node_degree[s] += 1
        if labels_np[s] == labels_np[d]:
            node_homophily[s] += 1
    
    # Avoid division by zero for isolated nodes
    mask = node_degree > 0
    node_homophily[mask] = node_homophily[mask] / node_degree[mask]
    
    return node_homophily


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name="Block"):
        self.name = name
        self.elapsed = 0
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        print(f"⏱  {self.name}: {self.elapsed:.3f}s")


def format_num(n):
    """Format large numbers with commas for readability."""
    return f"{n:,}"


def print_separator(char="─", length=60):
    """Print a visual separator line."""
    print(char * length)


def print_header(title):
    """Print a formatted section header."""
    print_separator("═")
    print(f"  {title}")
    print_separator("═")
