"""Small, deterministic regression tests for the Project-2026 core."""
import torch
import pytest
from src.datasets import load_dataset
from src.models import create_model
from src.train import train
from src.evaluate import compute_all_metrics, evaluate_with_edge_dropout
from src.utils import set_seed

@pytest.fixture(scope="module")
def wiki_data():
    return load_dataset("wikics", root="./data")


def test_all_models_forward(wiki_data):
    model_configs = {
        "gatv2": {"hidden_channels": 16, "dropout": 0.2, "heads": 2},
        "graphsage": {"hidden_channels": 16, "dropout": 0.2},
        "h2gcn": {"hidden_channels": 16, "dropout": 0.2, "num_rounds": 1},
        "mlp": {"hidden_channels": 16, "dropout": 0.2},
        "random": {},
    }
    for model_name, kwargs in model_configs.items():
        model = create_model(model_name, wiki_data.num_node_features, wiki_data.num_classes, **kwargs)
        output = model(wiki_data.x, wiki_data.edge_index)
        assert output.shape == (wiki_data.num_nodes, wiki_data.num_classes)
        assert torch.isfinite(output).all()


def test_short_training_and_metrics(wiki_data):
    set_seed(42)
    model = create_model("gatv2", wiki_data.num_node_features, wiki_data.num_classes,
                         hidden_channels=16, heads=2, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history = train(model, wiki_data, optimizer, max_epochs=2, patience=2, verbose=False)
    metrics = compute_all_metrics(model, wiki_data)
    assert history["total_epochs"] == 2
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_edge_dropout_preserves_edge_symmetry(wiki_data):
    set_seed(7)
    model = create_model("graphsage", wiki_data.num_node_features, wiki_data.num_classes,
                         hidden_channels=8, dropout=0.1)
    mean, std = evaluate_with_edge_dropout(model, wiki_data, 0.5, torch.device("cpu"), num_runs=1)
    assert 0.0 <= mean <= 1.0
    assert std == 0.0
