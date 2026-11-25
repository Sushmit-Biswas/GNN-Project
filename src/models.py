"""
GNN model architectures for the project.
Implements GATv2, GraphSAGE, H2GCN, MLP baseline, and Random baseline.
All models follow a unified interface for easy benchmarking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv
from torch_geometric.utils import degree, remove_self_loops, scatter


# ──────────────────────────────────────────────────────────────
#  Base Model
# ──────────────────────────────────────────────────────────────

class BaseGNN(nn.Module):
    """
    Base class for all GNN models.
    Provides a unified interface for training and evaluation.
    """
    
    def __init__(self):
        super().__init__()
        self.model_name = "BaseGNN"
    
    def reset_parameters(self):
        """Reset all learnable parameters (for multi-run experiments)."""
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    
    def get_embeddings(self, x, edge_index):
        """
        Extract node embeddings from the penultimate layer.
        Override in subclasses.
        """
        raise NotImplementedError
    
    def __repr__(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"{self.model_name}(\n"
            f"  Total params: {total:,}\n"
            f"  Trainable params: {trainable:,}\n"
            f")"
        )


# ──────────────────────────────────────────────────────────────
#  Model 1: GATv2 (Graph Attention Network v2)
# ──────────────────────────────────────────────────────────────

class GATv2Model(BaseGNN):
    """
    GATv2: Fixes the static attention limitation of vanilla GAT.
    
    Architecture:
      Input → GATv2Conv (multi-head) → ELU → Dropout 
            → GATv2Conv (single-head) → LogSoftmax
    
    Key innovation: Dynamic attention scoring — the attention function
    can represent a strictly wider class of functions than GAT.
    
    Paper: "How Attentive are Graph Attention Networks?" (Brody et al., 2022)
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=8, dropout=0.6, num_layers=2):
        super().__init__()
        self.model_name = "GATv2"
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer: multi-head attention
        self.convs.append(
            GATv2Conv(in_channels, hidden_channels, heads=heads, 
                      dropout=dropout, concat=True)
        )
        self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # Intermediate layers (if num_layers > 2)
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(hidden_channels * heads, hidden_channels, 
                          heads=heads, dropout=dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # Final layer: single-head, no concatenation
        self.convs.append(
            GATv2Conv(hidden_channels * heads, out_channels, 
                      heads=1, dropout=dropout, concat=False)
        )
    
    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, x, edge_index):
        """Extract embeddings from the penultimate layer."""
        with torch.no_grad():
            for i in range(self.num_layers - 1):
                x = self.convs[i](x, edge_index)
                x = self.norms[i](x)
                x = F.elu(x)
            return x


# ──────────────────────────────────────────────────────────────
#  Model 2: GraphSAGE (Sample and Aggregate)
# ──────────────────────────────────────────────────────────────

class GraphSAGEModel(BaseGNN):
    """
    GraphSAGE: Inductive Representation Learning on Large Graphs.
    
    Architecture:
      Input → SAGEConv → LayerNorm → ReLU → Dropout
            → SAGEConv → LogSoftmax
    
    Key innovation: Learns aggregation functions over sampled neighborhoods
    instead of using the full adjacency matrix. Uses mean/pool/LSTM
    aggregators. Supports inductive learning on unseen nodes.
    
    Paper: "Inductive Representation Learning on Large Graphs"
           (Hamilton et al., NeurIPS 2017)
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout=0.5, num_layers=2, aggr='mean'):
        super().__init__()
        self.model_name = "GraphSAGE"
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Final layer
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))
    
    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, x, edge_index):
        """Extract embeddings from the penultimate layer."""
        with torch.no_grad():
            for i in range(self.num_layers - 1):
                x = self.convs[i](x, edge_index)
                x = self.norms[i](x)
                x = F.relu(x)
            return x


# ──────────────────────────────────────────────────────────────
#  Model 3: H2GCN (Heterophily-aware GCN)
# ──────────────────────────────────────────────────────────────

class H2GCNModel(BaseGNN):
    """
    H2GCN: Beyond Homophily in Graph Neural Networks.
    
    Architecture:
      1. Feature projection: X → H_0 via linear layer
      2. Neighborhood aggregation: separate 1-hop and 2-hop embeddings
      3. Ego-neighbor separation: ego embedding is NOT mixed into
         the neighbor aggregation (key design for heterophily)
      4. Concatenation: [H_0 || H_1_hop || H_2_hop] → classifier
    
    Key innovations for heterophily:
      - Separates ego and neighbor representations
      - Uses higher-order (2-hop) neighborhoods to capture
        "similar nodes at distance 2" pattern common in heterophilic graphs
      - Combines intermediate representations via concatenation
    
    Paper: "Beyond Homophily in Graph Neural Networks: Current Limitations
           and Effective Designs" (Zhu et al., NeurIPS 2020)
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout=0.5, num_rounds=2):
        super().__init__()
        self.model_name = "H2GCN"
        self.dropout = dropout
        self.num_rounds = num_rounds
        
        # Feature projection
        self.proj = nn.Linear(in_channels, hidden_channels)
        self.proj_norm = nn.LayerNorm(hidden_channels)
        
        # Final classifier takes concatenated embeddings:
        # ego (hidden) + num_rounds * (1-hop hidden + 2-hop hidden)
        concat_dim = hidden_channels * (1 + 2 * num_rounds)
        self.classifier = nn.Linear(concat_dim, out_channels)
    
    def _aggregate(self, x, edge_index, num_nodes):
        """Aggregate neighbors without mixing the ego feature into them."""
        edge_index, _ = remove_self_loops(edge_index)
        row, col = edge_index[0], edge_index[1]
        deg = degree(row, num_nodes=num_nodes, dtype=x.dtype).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = scatter(x[col] * norm.unsqueeze(-1), row, dim=0,
                      dim_size=num_nodes, reduce='sum')
        return out
    
    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        
        # Step 1: project features (ego embedding)
        h_ego = self.proj(x)
        h_ego = self.proj_norm(h_ego)
        h_ego = F.relu(h_ego)
        h_ego = F.dropout(h_ego, p=self.dropout, training=self.training)
        
        # Step 2: iterative neighborhood aggregation
        parts = [h_ego]
        h_prev = h_ego
        for _ in range(self.num_rounds):
            # 1-hop aggregation
            h_1hop = self._aggregate(h_prev, edge_index, num_nodes)
            # 2-hop aggregation (aggregate the 1-hop result again)
            h_2hop = self._aggregate(h_1hop, edge_index, num_nodes)
            parts.extend([h_1hop, h_2hop])
            h_prev = h_1hop  # next round starts from 1-hop
        
        # Step 3: concatenate all representations
        h_cat = torch.cat(parts, dim=-1)
        h_cat = F.dropout(h_cat, p=self.dropout, training=self.training)
        
        # Step 4: classify
        out = self.classifier(h_cat)
        return F.log_softmax(out, dim=1)
    
    def get_embeddings(self, x, edge_index):
        """Extract concatenated representations before the classifier."""
        with torch.no_grad():
            num_nodes = x.size(0)
            h_ego = F.relu(self.proj_norm(self.proj(x)))
            parts = [h_ego]
            h_prev = h_ego
            for _ in range(self.num_rounds):
                h_1hop = self._aggregate(h_prev, edge_index, num_nodes)
                h_2hop = self._aggregate(h_1hop, edge_index, num_nodes)
                parts.extend([h_1hop, h_2hop])
                h_prev = h_1hop
            return torch.cat(parts, dim=-1)


# ──────────────────────────────────────────────────────────────
#  Baseline 1: MLP (Feature-Only, No Graph Structure)
# ──────────────────────────────────────────────────────────────

class MLPModel(BaseGNN):
    """
    MLP baseline: uses only node features, completely ignores graph structure.
    
    Architecture:
      Input → Linear → LayerNorm → ReLU → Dropout → Linear → LogSoftmax
    
    Purpose: Establishes how much signal the node features alone carry.
    If a GNN cannot beat the MLP, the graph structure is not helping
    (or the GNN is not exploiting it well).
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout=0.5, num_layers=2):
        super().__init__()
        self.model_name = "MLP"
        self.dropout = dropout
        
        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.norms.append(nn.LayerNorm(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        self.lins.append(nn.Linear(hidden_channels, out_channels))
    
    def forward(self, x, edge_index):
        # edge_index is accepted but ignored (unified interface)
        for i in range(len(self.lins) - 1):
            x = self.lins[i](x)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, x, edge_index):
        """Extract embeddings from the penultimate layer."""
        with torch.no_grad():
            for i in range(len(self.lins) - 1):
                x = self.lins[i](x)
                x = self.norms[i](x)
                x = F.relu(x)
            return x


# ──────────────────────────────────────────────────────────────
#  Baseline 2: Random (No Learning, Uniform Prediction)
# ──────────────────────────────────────────────────────────────

class RandomModel(BaseGNN):
    """
    Random baseline: outputs uniform random log-probabilities.
    
    Purpose: Provides a performance floor. Any model that cannot beat
    random assignment is not learning anything useful. Expected accuracy
    is ~1/num_classes.
    """
    
    def __init__(self, in_channels, hidden_channels=0, out_channels=None, **kwargs):
        super().__init__()
        self.model_name = "Random"
        self.out_channels = out_channels
        # A dummy parameter so optimizers don't complain about empty param list
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
    
    def forward(self, x, edge_index):
        logits = torch.randn(x.size(0), self.out_channels, device=x.device)
        return F.log_softmax(logits, dim=1)
    
    def get_embeddings(self, x, edge_index):
        """Random model has no learned representations."""
        return torch.randn(x.size(0), self.out_channels, device=x.device)


# ──────────────────────────────────────────────────────────────
#  Model Factory
# ──────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    'gatv2': GATv2Model,
    'graphsage': GraphSAGEModel,
    'h2gcn': H2GCNModel,
    'mlp': MLPModel,
    'random': RandomModel,
}


def create_model(model_name, in_channels, num_classes, **kwargs):
    """
    Factory function to create a model by name.
    
    Args:
        model_name: one of 'gatv2', 'graphsage', 'h2gcn', 'mlp', 'random'
        in_channels: number of input features
        num_classes: number of output classes
        **kwargs: model-specific hyperparameters
    
    Returns:
        model: instantiated GNN model
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from {list(MODEL_REGISTRY.keys())}"
        )
    
    # Default hyperparameters per model
    defaults = {
        'gatv2': {'hidden_channels': 64, 'heads': 8, 'dropout': 0.6},
        'graphsage': {'hidden_channels': 64, 'dropout': 0.5},
        'h2gcn': {'hidden_channels': 64, 'dropout': 0.5, 'num_rounds': 2},
        'mlp': {'hidden_channels': 64, 'dropout': 0.5},
        'random': {},
    }
    
    # Merge defaults with provided kwargs
    params = {**defaults[model_name], **kwargs}
    
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(
        in_channels=in_channels,
        out_channels=num_classes,
        **params
    )
    
    return model


def get_all_model_names():
    """Return list of all model names."""
    return list(MODEL_REGISTRY.keys())
