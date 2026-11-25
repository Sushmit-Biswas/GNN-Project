"""
Dataset loading and preprocessing for the GNN project.
Provides a unified interface for WikiCS, Amazon Computers, and Roman-empire.
"""

import torch
import numpy as np
from torch_geometric.datasets import WikiCS, Amazon, HeterophilousGraphDataset
from torch_geometric.transforms import NormalizeFeatures, ToUndirected
from torch_geometric.utils import degree
from sklearn.model_selection import train_test_split

from .utils import compute_homophily, format_num, print_separator, print_header


# ──────────────────────────────────────────────────────────────
#  Dataset Registry
# ──────────────────────────────────────────────────────────────

DATASET_INFO = {
    'wikics': {
        'name': 'WikiCS',
        'domain': 'Wikipedia Computer Science',
        'node_semantics': 'Wikipedia articles about Computer Science topics',
        'edge_semantics': 'Hyperlinks between articles (undirected)',
        'feature_semantics': '300-dim GloVe embeddings of article text',
        'class_semantics': '10 CS sub-fields (e.g., ML, NLP, Computer Vision)',
        'homophily_level': 'Medium (~0.65)',
    },
    'amazon-computers': {
        'name': 'Amazon Computers',
        'domain': 'E-commerce Co-Purchase',
        'node_semantics': 'Computer products on Amazon',
        'edge_semantics': 'Co-purchase links (frequently bought together)',
        'feature_semantics': '767-dim bag-of-words from product reviews',
        'class_semantics': '10 product categories (Laptops, Keyboards, etc.)',
        'homophily_level': 'High (~0.78)',
    },
    'roman-empire': {
        'name': 'Roman-empire',
        'domain': 'Wikipedia Word Dependency',
        'node_semantics': 'Words from the Wikipedia article about the Roman Empire',
        'edge_semantics': 'Word adjacency + syntactic dependency tree edges',
        'feature_semantics': '300-dim FastText word embeddings',
        'class_semantics': '18 syntactic dependency roles (nsubj, dobj, amod, etc.)',
        'homophily_level': 'Very Low / Heterophilic (~0.05)',
    },
}


# ──────────────────────────────────────────────────────────────
#  Loading Functions
# ──────────────────────────────────────────────────────────────

def load_wikics(root='./data'):
    """
    Load WikiCS dataset with standard preprocessing.
    Uses the first of 20 pre-defined train/val splits.
    """
    dataset = WikiCS(root=f'{root}/WikiCS', is_undirected=True)
    data = dataset[0]
    
    # WikiCS provides 20 splits — use the first one
    data.train_mask = data.train_mask[:, 0]
    data.val_mask = data.val_mask[:, 0]
    # test_mask is a single mask already
    if data.test_mask.dim() > 1:
        data.test_mask = data.test_mask[:, 0]
    
    # Drop the stopping_mask (not needed for our experiments)
    if hasattr(data, 'stopping_mask'):
        del data.stopping_mask
    
    data.dataset_name = 'WikiCS'
    data.num_classes = dataset.num_classes
    return data


def load_amazon_computers(root='./data'):
    """
    Load Amazon Computers dataset.
    Creates stratified 60/20/20 train/val/test splits (no pre-defined splits).
    """
    dataset = Amazon(root=f'{root}/Amazon', name='Computers')
    data = dataset[0]
    
    # Create stratified splits
    num_nodes = data.num_nodes
    labels = data.y.numpy()
    indices = np.arange(num_nodes)
    
    # 60% train, 20% val, 20% test (stratified by class)
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.4, stratify=labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=42
    )
    
    # Create boolean masks
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
    
    data.dataset_name = 'Amazon Computers'
    data.num_classes = dataset.num_classes
    return data


def load_roman_empire(root='./data'):
    """
    Load Roman-empire heterophilous dataset.
    The dataset comes with pre-defined train/val/test splits.
    Applies ToUndirected transform (edges stored once in the raw data).
    """
    transform = ToUndirected()
    dataset = HeterophilousGraphDataset(
        root=f'{root}/HeterophilousGraph', 
        name='Roman-empire',
        transform=transform
    )
    data = dataset[0]

    if data.train_mask.dim() > 1:
        data.train_mask = data.train_mask[:, 0]
    if data.val_mask.dim() > 1:
        data.val_mask = data.val_mask[:, 0]
    if data.test_mask.dim() > 1:
        data.test_mask = data.test_mask[:, 0]
    
    data.dataset_name = 'Roman-empire'
    data.num_classes = dataset.num_classes
    return data


# ──────────────────────────────────────────────────────────────
#  Unified Loader
# ──────────────────────────────────────────────────────────────

def load_dataset(name, root='./data'):
    """
    Load a dataset by name with unified interface.
    
    Args:
        name: one of 'wikics', 'amazon-computers', 'roman-empire'
        root: root directory for downloaded data
    
    Returns:
        data: PyG Data object with train_mask, val_mask, test_mask
    """
    loaders = {
        'wikics': load_wikics,
        'amazon-computers': load_amazon_computers,
        'roman-empire': load_roman_empire,
    }
    
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders.keys())}")
    
    data = loaders[name](root)
    return data


def get_all_dataset_names():
    """Return list of all dataset names."""
    return ['wikics', 'amazon-computers', 'roman-empire']


# ──────────────────────────────────────────────────────────────
#  Dataset Statistics & Summary
# ──────────────────────────────────────────────────────────────

def print_dataset_summary(data, name=None):
    """
    Print a comprehensive summary of a dataset.
    
    Args:
        data: PyG Data object
        name: dataset key (for semantic info lookup)
    """
    dataset_name = getattr(data, 'dataset_name', name or 'Unknown')
    print_header(f"Dataset: {dataset_name}")
    
    # Basic statistics
    print(f"  Nodes:          {format_num(data.num_nodes)}")
    print(f"  Edges:          {format_num(data.num_edges)}")
    print(f"  Node Features:  {data.num_node_features}")
    print(f"  Classes:        {data.num_classes}")
    
    # Density uses the number of ordered edge entries exposed by PyG.
    n = data.num_nodes
    e = data.num_edges
    density = e / (n * (n - 1)) if n > 1 else 0.0
    print(f"  Density:        {density:.6f}")
    
    # Average degree
    deg = degree(data.edge_index[0], num_nodes=n)
    print(f"  Avg Degree:     {deg.mean().item():.2f}")
    print(f"  Max Degree:     {deg.max().item():.0f}")
    print(f"  Min Degree:     {deg.min().item():.0f}")
    
    # Homophily
    if data.y is not None:
        h = compute_homophily(data.edge_index, data.y)
        print(f"  Edge Homophily: {h:.4f}")
    
    # Split sizes
    if hasattr(data, 'train_mask'):
        print(f"\n  Split Sizes:")
        print(f"    Train: {data.train_mask.sum().item():,} ({data.train_mask.sum().item()/n*100:.1f}%)")
        print(f"    Val:   {data.val_mask.sum().item():,} ({data.val_mask.sum().item()/n*100:.1f}%)")
        print(f"    Test:  {data.test_mask.sum().item():,} ({data.test_mask.sum().item()/n*100:.1f}%)")
    
    # Class distribution
    if data.y is not None:
        print(f"\n  Class Distribution:")
        classes, counts = torch.unique(data.y, return_counts=True)
        for c, cnt in zip(classes.tolist(), counts.tolist()):
            bar = "█" * int(cnt / counts.max().item() * 20)
            print(f"    Class {c:2d}: {cnt:6,} {bar}")
    
    # Semantic info (if available)
    if name and name in DATASET_INFO:
        info = DATASET_INFO[name]
        print(f"\n  Semantics:")
        print(f"    Nodes:    {info['node_semantics']}")
        print(f"    Edges:    {info['edge_semantics']}")
        print(f"    Features: {info['feature_semantics']}")
        print(f"    Classes:  {info['class_semantics']}")
        print(f"    Homophily Level: {info['homophily_level']}")
    
    print_separator()


def get_degree_distribution(data):
    """
    Compute degree distribution for visualization.
    
    Returns:
        degrees: numpy array of node degrees
    """
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    return deg.numpy()
