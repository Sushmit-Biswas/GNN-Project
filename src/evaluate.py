"""
Evaluation metrics, confusion matrix generation, and embedding extraction.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix as sk_confusion_matrix
)


# ──────────────────────────────────────────────────────────────
#  Core Evaluation
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, data, mask=None):
    """
    Evaluate model on a specific split.
    
    Args:
        model: trained GNN model
        data: PyG Data object
        mask: boolean mask for the split (default: test_mask)
    
    Returns:
        dict with accuracy and predictions
    """
    model.eval()
    
    if mask is None:
        mask = data.test_mask
    
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1)
    true = data.y[mask]
    
    correct = (pred == true).sum().item()
    total = mask.sum().item()
    accuracy = correct / total
    
    return {
        'accuracy': accuracy,
        'predictions': pred.cpu(),
        'true_labels': true.cpu(),
        'logits': out[mask].cpu(),
    }


@torch.no_grad()
def compute_all_metrics(model, data, mask=None):
    """
    Compute comprehensive metrics for a model on a data split.
    
    Returns:
        dict with accuracy, macro_f1, weighted_f1, per_class_f1, 
        classification_report, confusion_matrix
    """
    eval_result = evaluate(model, data, mask)
    
    pred = eval_result['predictions'].numpy()
    true = eval_result['true_labels'].numpy()
    
    metrics = {
        'accuracy': eval_result['accuracy'],
        'macro_f1': f1_score(true, pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(true, pred, average='weighted', zero_division=0),
        'per_class_f1': f1_score(true, pred, average=None, zero_division=0),
        'confusion_matrix': sk_confusion_matrix(true, pred),
        'predictions': pred,
        'true_labels': true,
        'logits': eval_result['logits'].numpy(),
    }
    
    return metrics


def print_classification_report(model, data, mask=None, target_names=None):
    """Print a detailed sklearn classification report."""
    eval_result = evaluate(model, data, mask)
    pred = eval_result['predictions'].numpy()
    true = eval_result['true_labels'].numpy()
    
    report = classification_report(
        true, pred, 
        target_names=target_names,
        zero_division=0
    )
    print(report)
    return report


# ──────────────────────────────────────────────────────────────
#  Embedding Extraction (for t-SNE)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, data):
    """
    Extract node embeddings from the penultimate layer of the model.
    
    Args:
        model: trained GNN model (must implement get_embeddings())
        data: PyG Data object
    
    Returns:
        embeddings: numpy array of shape [N, hidden_dim]
    """
    model.eval()
    embeddings = model.get_embeddings(data.x, data.edge_index)
    return embeddings.cpu().numpy()


@torch.no_grad()
def extract_final_embeddings(model, data):
    """
    Extract embeddings from the final layer (before softmax).
    Useful for visualizing the decision space.
    
    Returns:
        embeddings: numpy array of shape [N, num_classes]
    """
    model.eval()
    out = model(data.x, data.edge_index)
    return out.cpu().numpy()


# ──────────────────────────────────────────────────────────────
#  Robustness Evaluation Helpers
# ──────────────────────────────────────────────────────────────

def evaluate_with_edge_dropout(model, data, drop_rate, device, num_runs=5):
    """
    Evaluate model accuracy after randomly removing a fraction of edges.
    
    Args:
        model: trained GNN model
        data: PyG Data object
        drop_rate: fraction of edges to remove (0.0 to 1.0)
        device: torch device
        num_runs: number of random dropout runs to average
    
    Returns:
        mean_acc: average accuracy over runs
        std_acc: standard deviation
    """
    model.eval()
    accs = []
    
    for _ in range(num_runs):
        # Create edge mask
        edge_index = data.edge_index.cpu()
        undirected_pairs = torch.stack(
            [torch.minimum(edge_index[0], edge_index[1]),
             torch.maximum(edge_index[0], edge_index[1])], dim=0
        )
        _, inverse = torch.unique(undirected_pairs, dim=1, return_inverse=True)
        pair_keep = torch.rand(int(inverse.max().item()) + 1) > drop_rate
        keep_mask = pair_keep[inverse]
        
        # Apply dropout
        edge_index_dropped = data.edge_index[:, keep_mask].to(device)
        
        with torch.no_grad():
            out = model(data.x.to(device), edge_index_dropped)
            pred = out[data.test_mask].argmax(dim=1)
            true = data.y[data.test_mask].to(device)
            acc = (pred == true).float().mean().item()
            accs.append(acc)
    
    return np.mean(accs), np.std(accs)


def evaluate_with_feature_noise(model, data, noise_std, device, num_runs=5):
    """
    Evaluate model accuracy after adding Gaussian noise to features.
    
    Args:
        model: trained GNN model
        data: PyG Data object
        noise_std: standard deviation of Gaussian noise
        device: torch device
        num_runs: number of random noise runs to average
    
    Returns:
        mean_acc: average accuracy over runs
        std_acc: standard deviation
    """
    model.eval()
    accs = []
    
    for _ in range(num_runs):
        # Add noise to features
        noise = torch.randn_like(data.x) * noise_std
        x_noisy = (data.x + noise).to(device)
        
        with torch.no_grad():
            out = model(x_noisy, data.edge_index.to(device))
            pred = out[data.test_mask].argmax(dim=1)
            true = data.y[data.test_mask].to(device)
            acc = (pred == true).float().mean().item()
            accs.append(acc)
    
    return np.mean(accs), np.std(accs)


def evaluate_with_label_scarcity(model_name, data, device, 
                                  train_fraction, seed=42, 
                                  max_epochs=300, patience=30,
                                  lr=0.005, weight_decay=5e-4,
                                  **model_kwargs):
    """
    Train and evaluate a model using only a fraction of the training labels.
    
    Args:
        model_name: model name string
        data: PyG Data object
        device: torch device
        train_fraction: fraction of training nodes to keep (0.0 to 1.0)
        seed: random seed
    
    Returns:
        test_acc: test accuracy with reduced training set
    """
    from .train import train as train_fn
    from .models import create_model
    from .utils import set_seed
    
    set_seed(seed)
    
    # Create reduced training mask
    train_indices = data.train_mask.nonzero(as_tuple=True)[0]
    num_keep = max(1, int(len(train_indices) * train_fraction))
    
    # Keep every class represented whenever the requested fraction permits it.
    selected = []
    generator = torch.Generator().manual_seed(seed)
    for label in torch.unique(data.y[train_indices]):
        class_indices = train_indices[data.y[train_indices] == label]
        selected.append(class_indices[torch.randperm(len(class_indices), generator=generator)[:1]])
    selected = torch.cat(selected)
    remaining = train_indices[~torch.isin(train_indices, selected)]
    extra = max(0, num_keep - len(selected))
    if extra:
        remaining = remaining[torch.randperm(len(remaining), generator=generator)[:extra]]
        selected = torch.cat([selected, remaining])
    reduced_mask = torch.zeros_like(data.train_mask)
    reduced_mask[selected] = True
    
    # Create modified data
    import copy
    data_reduced = copy.deepcopy(data)
    data_reduced.train_mask = reduced_mask
    
    # Train model
    model = create_model(
        model_name,
        in_channels=data.num_node_features,
        num_classes=data.num_classes,
        **model_kwargs
    ).to(device)
    
    data_dev = data_reduced.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_fn(model, data_dev, optimizer, max_epochs=max_epochs, 
            patience=patience, verbose=False)
    
    # Evaluate on test set
    metrics = compute_all_metrics(model, data_dev)
    return metrics['accuracy'], metrics['macro_f1']
