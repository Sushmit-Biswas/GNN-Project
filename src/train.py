"""
Training loop with early stopping, hyperparameter tuning, and logging.
"""

import time
import copy
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .models import create_model
from .evaluate import evaluate, compute_all_metrics
from .utils import set_seed, count_parameters, print_header, print_separator


# ──────────────────────────────────────────────────────────────
#  Single Training Run
# ──────────────────────────────────────────────────────────────

def train_epoch(model, data, optimizer):
    """
    Train for one epoch.
    
    Returns:
        loss: training loss (float)
    """
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def train(model, data, optimizer, max_epochs=500, patience=50, verbose=True):
    """
    Full training loop with early stopping on validation accuracy.
    
    Args:
        model: GNN model
        data: PyG Data object (with train_mask, val_mask, test_mask)
        optimizer: torch optimizer
        max_epochs: maximum number of epochs
        patience: early stopping patience
        verbose: whether to print progress
    
    Returns:
        dict: training history with losses, accuracies, and best epoch
    """
    best_val_acc = -1.0
    best_model_state = None
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_loss': [],
        'best_epoch': 0,
        'total_epochs': 0,
        'training_time': 0,
    }
    
    start_time = time.perf_counter()
    
    epoch = 0
    epoch_iter = range(1, max_epochs + 1)
    if verbose:
        epoch_iter = tqdm(epoch_iter, desc=f"Training {model.model_name}", 
                          leave=False, ncols=80)
    
    for epoch in epoch_iter:
        # Train
        train_loss = train_epoch(model, data, optimizer)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            
            # Training metrics
            train_pred = out[data.train_mask].argmax(dim=1)
            train_acc = (train_pred == data.y[data.train_mask]).float().mean().item()
            
            # Validation metrics
            val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask]).item()
            val_pred = out[data.val_mask].argmax(dim=1)
            val_acc = (val_pred == data.y[data.val_mask]).float().mean().item()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            history['best_epoch'] = epoch
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best val_acc: {best_val_acc:.4f} at epoch {history['best_epoch']})")
            break
        
        # Update progress bar
        if verbose and hasattr(epoch_iter, 'set_postfix'):
            epoch_iter.set_postfix({
                'loss': f"{train_loss:.3f}",
                'val_acc': f"{val_acc:.3f}",
                'best': f"{best_val_acc:.3f}",
            })
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    history['total_epochs'] = epoch
    history['training_time'] = time.perf_counter() - start_time
    history['best_val_acc'] = best_val_acc
    
    return history


# ──────────────────────────────────────────────────────────────
#  Multi-Seed Training (for mean ± std)
# ──────────────────────────────────────────────────────────────

def train_multi_seed(model_name, data, device, seeds=[42, 123, 456, 789, 1024],
                     lr=0.005, weight_decay=5e-4, max_epochs=500, patience=50,
                     verbose=True, **model_kwargs):
    """
    Train a model multiple times with different seeds and report mean ± std.
    
    Returns:
        results: dict with mean/std metrics and per-seed details
    """
    all_test_metrics = []
    all_histories = []
    
    for i, seed in enumerate(seeds):
        set_seed(seed)
        
        # Create fresh model for each seed
        model = create_model(
            model_name,
            in_channels=data.num_node_features,
            num_classes=data.num_classes,
            **model_kwargs
        ).to(device)
        
        data_dev = data.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        
        if verbose:
            print(f"\n  Run {i+1}/{len(seeds)} (seed={seed})")
        
        # Train
        history = train(model, data_dev, optimizer, max_epochs=max_epochs,
                       patience=patience, verbose=verbose)
        
        # Test evaluation
        test_metrics = compute_all_metrics(model, data_dev)
        all_test_metrics.append(test_metrics)
        all_histories.append(history)
        
        if verbose:
            print(f"    Test Acc: {test_metrics['accuracy']:.4f}, "
                  f"F1: {test_metrics['macro_f1']:.4f}")
    
    # Aggregate results
    accs = [m['accuracy'] for m in all_test_metrics]
    f1s = [m['macro_f1'] for m in all_test_metrics]
    times = [h['training_time'] for h in all_histories]
    
    results = {
        'model_name': model_name,
        'dataset_name': getattr(data, 'dataset_name', 'Unknown'),
        'mean_accuracy': np.mean(accs),
        'std_accuracy': np.std(accs),
        'mean_f1': np.mean(f1s),
        'std_f1': np.std(f1s),
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'per_seed_metrics': all_test_metrics,
        'per_seed_histories': all_histories,
        'num_seeds': len(seeds),
    }
    
    if verbose:
        print(f"\n  ──────────────────────────────────")
        print(f"  {model_name} on {results['dataset_name']}:")
        print(f"    Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        print(f"    Macro F1: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
        print(f"    Avg Time: {results['mean_time']:.2f}s ± {results['std_time']:.2f}s")
    
    return results


# ──────────────────────────────────────────────────────────────
#  Hyperparameter Tuning
# ──────────────────────────────────────────────────────────────

def tune_hyperparameters(model_name, data, device,
                         param_grid=None, max_epochs=300, patience=30,
                         verbose=True):
    """
    Grid search over hyperparameters for a model-dataset pair.
    Uses validation accuracy to select the best configuration.
    
    Args:
        model_name: model name string
        data: PyG Data object
        device: torch device
        param_grid: dict of param_name → list of values to try
        max_epochs: max epochs per trial
        patience: early stopping patience per trial
    
    Returns:
        best_params: best hyperparameter configuration
        best_val_acc: best validation accuracy achieved
        all_results: list of (params, val_acc) tuples
    """
    if model_name == 'random':
        return {}, 0.0, []

    if param_grid is None:
        param_grid = get_default_param_grid(model_name)
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    if verbose:
        print_header(f"Tuning {model_name} on {getattr(data, 'dataset_name', '?')}")
        print(f"  Search space: {len(combinations)} combinations")
        print(f"  Parameters: {param_names}")
    
    best_val_acc = 0
    best_params = None
    all_results = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        
        # Separate optimizer params from model params
        lr = params.pop('lr', 0.005)
        weight_decay = params.pop('weight_decay', 5e-4)
        
        set_seed(42)
        
        try:
            model = create_model(
                model_name,
                in_channels=data.num_node_features,
                num_classes=data.num_classes,
                **params
            ).to(device)
            
            data_dev = data.to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            
            history = train(model, data_dev, optimizer, max_epochs=max_epochs,
                           patience=patience, verbose=False)
            
            val_acc = history['best_val_acc']
            
            # Restore optimizer params for recording
            params['lr'] = lr
            params['weight_decay'] = weight_decay
            all_results.append((params.copy(), val_acc))
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = params.copy()
            
            if verbose:
                status = "★" if val_acc == best_val_acc else " "
                print(f"  {status} [{i+1}/{len(combinations)}] "
                      f"val_acc={val_acc:.4f} | {params}")
                
        except Exception as e:
            if verbose:
                print(f"  ✗ [{i+1}/{len(combinations)}] FAILED: {e} | {params}")
            params['lr'] = lr
            params['weight_decay'] = weight_decay
            all_results.append((params.copy(), 0.0))
    
    if verbose:
        print_separator()
        print(f"  Best val_acc: {best_val_acc:.4f}")
        print(f"  Best params:  {best_params}")
        print_separator()
    
    return best_params, best_val_acc, all_results


def get_default_param_grid(model_name):
    """Get default hyperparameter search grid for a model."""
    
    common = {
        'lr': [0.001, 0.005, 0.01],
        'weight_decay': [0, 5e-4, 1e-3],
        'dropout': [0.3, 0.5, 0.7],
    }
    
    model_specific = {
        'gatv2': {
            'hidden_channels': [32, 64],
            'heads': [4, 8],
        },
        'graphsage': {
            'hidden_channels': [32, 64, 128],
        },
        'h2gcn': {
            'hidden_channels': [32, 64, 128],
            'num_rounds': [1, 2],
        },
        'mlp': {
            'hidden_channels': [32, 64, 128],
        },
    }
    
    grid = {**common, **model_specific.get(model_name, {})}
    return grid


def get_quick_param_grid(model_name):
    """
    Smaller grid for quick experiments (e.g., CPU-only or debugging).
    """
    common = {
        'lr': [0.005, 0.01],
        'weight_decay': [5e-4],
        'dropout': [0.5],
    }
    
    model_specific = {
        'gatv2': {
            'hidden_channels': [64],
            'heads': [8],
        },
        'graphsage': {
            'hidden_channels': [64],
        },
        'h2gcn': {
            'hidden_channels': [64],
            'num_rounds': [2],
        },
        'mlp': {
            'hidden_channels': [64],
        },
    }
    
    grid = {**common, **model_specific.get(model_name, {})}
    return grid
