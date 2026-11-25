"""Reproducible command-line runner for Project-2026 (works on CPU, NVIDIA CUDA GPU, and Google Colab)."""
from __future__ import annotations

import argparse
import inspect
import itertools
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.manifold import TSNE

from src.datasets import load_dataset
from src.evaluate import (
    compute_all_metrics,
    evaluate_with_edge_dropout,
    evaluate_with_feature_noise,
    evaluate_with_label_scarcity,
)
from src.models import create_model, get_all_model_names
from src.train import train
from src.utils import count_parameters, set_seed

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
PLOTS = RESULTS / "plots"
MODELS = RESULTS / "models"

DATASET_ORDER = ["amazon-computers", "wikics", "roman-empire"]

MODEL_LABELS = {
    "gatv2": "GATv2",
    "graphsage": "GraphSAGE",
    "h2gcn": "H2GCN",
    "mlp": "MLP",
    "random": "Random",
}

TRAINABLE_MODELS = ["gatv2", "graphsage", "h2gcn", "mlp"]

PLOT_COLORS = {
    "GATv2": "#1b4965",
    "GraphSAGE": "#2a9d8f",
    "H2GCN": "#e76f51",
    "MLP": "#ca6702",
    "Random": "#6c757d",
}


def load_config():
    with (ROOT / "configs" / "hyperparams.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(requested_device: str | None, cfg: dict) -> torch.device:
    device_name = requested_device or cfg["global"].get("device", "auto")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested via --device cuda, but torch.cuda.is_available() is False.")
    use_cuda = device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"Using GPU: {torch.cuda.get_device_name(0)} ({props.total_memory / 1e9:.1f} GB VRAM)")
    else:
        print("Using CPU")
    return device


def make_tsne(perplexity: int = 30, max_iter: int = 1000, random_state: int = 42) -> TSNE:
    """Create TSNE compatible with both older (n_iter) and newer (max_iter) scikit-learn."""
    sig = inspect.signature(TSNE.__init__).parameters
    kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "random_state": random_state,
        "init": "pca",
    }
    if "max_iter" in sig:
        kwargs["max_iter"] = max_iter
    elif "n_iter" in sig:
        kwargs["n_iter"] = max_iter
    return TSNE(**kwargs)


def model_kwargs(model_name: str, config: dict, quick: bool, overrides: dict | None = None) -> dict:
    params = dict(config["models"][model_name])
    params.update(overrides or {})
    params.pop("lr", None)
    params.pop("weight_decay", None)
    if quick:
        params["hidden_channels"] = min(params.get("hidden_channels", 64), 32)
        if "heads" in params:
            params["heads"] = min(params["heads"], 4)
    return params


def tune_pair(model_name: str, data, device: torch.device, cfg: dict, quick: bool, skip_tuning: bool = False):
    """Select hyperparameters using validation accuracy only (never test_mask)."""
    defaults = dict(cfg["models"][model_name])
    if model_name == "random":
        return {}, []
    if skip_tuning:
        selected = {
            k: defaults[k]
            for k in ("lr", "weight_decay", "dropout", "hidden_channels")
            if k in defaults
        }
        return selected, []

    grid = cfg["tuning"]["quick" if quick else "full"]
    lrs = grid.get("lr", [defaults.get("lr", 0.005)])
    wds = grid.get("weight_decay", [defaults.get("weight_decay", 5e-4)])
    dropouts = grid.get("dropout", [defaults.get("dropout", 0.5)])
    hiddens = grid.get("hidden_channels", [defaults.get("hidden_channels", 64)])
    tune_epochs = int(grid.get("max_epochs", 15 if quick else 150))
    tune_patience = int(grid.get("patience", 5 if quick else 20))

    combos = list(itertools.product(lrs, wds, dropouts, hiddens))
    best_score = -1.0
    best = {}
    trial_rows = []
    data_device = data.to(device)

    for lr, wd, dropout, hidden in combos:
        candidate = {
            "lr": float(lr),
            "weight_decay": float(wd),
            "dropout": float(dropout),
            "hidden_channels": int(hidden),
        }
        set_seed(42)
        kwargs = model_kwargs(model_name, cfg, quick, candidate)
        model = create_model(model_name, data.num_node_features, data.num_classes, **kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=candidate["lr"], weight_decay=candidate["weight_decay"])
        history = train(model, data_device, optimizer, max_epochs=tune_epochs, patience=tune_patience, verbose=False)
        val_acc = float(history["best_val_acc"])
        trial_rows.append({
            "dataset": data.dataset_name,
            "model": MODEL_LABELS[model_name],
            "model_key": model_name,
            **candidate,
            "val_accuracy": val_acc,
            "epochs": history["total_epochs"],
        })
        if val_acc > best_score:
            best_score = val_acc
            best = candidate.copy()

    print(
        f"  Tuned {MODEL_LABELS[model_name]} on {data.dataset_name} "
        f"({len(combos)} configs) -> val_acc={best_score:.4f} | {best}"
    )
    return best, trial_rows


def compute_macro_f1(prediction: torch.Tensor, true: torch.Tensor) -> float:
    classes = torch.unique(true)
    f1_values = []
    for label in classes:
        tp = ((prediction == label) & (true == label)).sum().item()
        fp = ((prediction == label) & (true != label)).sum().item()
        fn = ((prediction != label) & (true == label)).sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1_values))


def train_one(model_name: str, data, device: torch.device, cfg: dict, seed: int, quick: bool, overrides: dict | None = None):
    set_seed(seed)
    params = cfg["models"][model_name]
    selected = dict(overrides or {})
    kwargs = model_kwargs(model_name, cfg, quick, selected)
    model = create_model(model_name, data.num_node_features, data.num_classes, **kwargs).to(device)
    data_device = data.to(device)

    if model_name == "random":
        model.eval()
        with torch.no_grad():
            output = model(data_device.x, data_device.edge_index)
            mask = data_device.test_mask
            prediction = output[mask].argmax(dim=1)
            true = data_device.y[mask]
            accuracy = (prediction == true).float().mean().item()
            macro_f1 = compute_macro_f1(prediction, true)
        return {
            "dataset": data.dataset_name,
            "model": MODEL_LABELS[model_name],
            "model_key": model_name,
            "seed": seed,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "best_val_accuracy": 0.0,
            "epochs": 0,
            "seconds": 0.0,
            "parameters": 0,
            "checkpoint": "",
            "selected_lr": None,
            "selected_weight_decay": None,
            "selected_dropout": None,
            "selected_hidden_channels": None,
        }, model, data_device, None

    lr = float(selected.get("lr", params.get("lr", 0.005)))
    weight_decay = float(selected.get("weight_decay", params.get("weight_decay", 5e-4)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = 30 if quick else int(cfg["global"]["max_epochs"])
    patience = 8 if quick else int(cfg["global"]["patience"])
    start = time.perf_counter()
    history = train(model, data_device, optimizer, max_epochs=epochs, patience=patience, verbose=False)
    elapsed = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        output = model(data_device.x, data_device.edge_index)
        mask = data_device.test_mask
        prediction = output[mask].argmax(dim=1)
        true = data_device.y[mask]
        accuracy = (prediction == true).float().mean().item()
        macro_f1 = compute_macro_f1(prediction, true)

    checkpoint = {
        "model_state": model.state_dict(),
        "model_name": model_name,
        "dataset": data.dataset_name,
        "seed": seed,
        "model_kwargs": kwargs,
        "lr": lr,
        "weight_decay": weight_decay,
        "best_val_accuracy": history["best_val_acc"],
    }
    checkpoint_path = MODELS / data.dataset_name.lower().replace(" ", "_") / model_name / f"seed_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

    return {
        "dataset": data.dataset_name,
        "model": MODEL_LABELS[model_name],
        "model_key": model_name,
        "seed": seed,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "best_val_accuracy": history["best_val_acc"],
        "epochs": history["total_epochs"],
        "seconds": elapsed,
        "parameters": count_parameters(model)[1],
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "selected_lr": lr,
        "selected_weight_decay": weight_decay,
        "selected_dropout": float(kwargs.get("dropout", 0.0)),
        "selected_hidden_channels": int(kwargs.get("hidden_channels", 0)),
    }, model, data_device, history


def save_benchmark_plot(summary: pd.DataFrame):
    model_order = [MODEL_LABELS[m] for m in get_all_model_names()]
    dataset_order = ["Amazon Computers", "WikiCS", "Roman-empire"]
    colors = [PLOT_COLORS[m] for m in model_order]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for metric, std_col, axis, title in [
        ("accuracy", "accuracy_std", axes[0], "Test Accuracy (mean ± std)"),
        ("macro_f1", "macro_f1_std", axes[1], "Macro-F1 (mean ± std)"),
    ]:
        pivot_mean = summary.pivot(index="dataset", columns="model", values=metric).reindex(
            index=dataset_order, columns=model_order
        )
        pivot_std = summary.pivot(index="dataset", columns="model", values=std_col).reindex(
            index=dataset_order, columns=model_order
        )
        pivot_mean.plot(
            kind="bar",
            yerr=pivot_std,
            ax=axis,
            width=0.78,
            color=colors,
            capsize=3,
            edgecolor="white",
            error_kw={"elinewidth": 1.0},
        )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("Score")
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="", fontsize=8, loc="upper right")
        axis.tick_params(axis="x", rotation=15)

    fig.savefig(PLOTS / "benchmark_comparison.png", dpi=180)
    plt.close(fig)


def save_training_curves(histories: dict):
    """Plot train/val accuracy curves (first seed) across datasets and trainable models."""
    datasets_present = [d for d in ["Amazon Computers", "WikiCS", "Roman-empire"] if d in histories]
    if not datasets_present:
        return
    fig, axes = plt.subplots(len(TRAINABLE_MODELS), len(datasets_present), figsize=(16, 12), constrained_layout=True)
    for i, mkey in enumerate(TRAINABLE_MODELS):
        label = MODEL_LABELS[mkey]
        color = PLOT_COLORS[label]
        for j, dlabel in enumerate(datasets_present):
            ax = axes[i][j]
            hist = histories[dlabel].get(mkey)
            if hist is None:
                ax.axis("off")
                continue
            epochs = range(1, len(hist["train_acc"]) + 1)
            ax.plot(epochs, hist["train_acc"], linestyle="--", color=color, alpha=0.55, label="Train Acc")
            ax.plot(epochs, hist["val_acc"], linewidth=2.0, color=color, label="Val Acc")
            if hist["best_epoch"] > 0:
                ax.axvline(hist["best_epoch"], color="gray", linestyle=":", alpha=0.7)
            ax.set_title(f"{label} — {dlabel}", fontsize=10, fontweight="bold")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
            if i == len(TRAINABLE_MODELS) - 1:
                ax.set_xlabel("Epoch")
            if j == 0:
                ax.set_ylabel("Accuracy")
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Training & Validation Accuracy Curves (Seed 42)", fontsize=14, fontweight="bold")
    fig.savefig(PLOTS / "training_curves.png", dpi=180)
    plt.close(fig)


def save_visualisation(model_records: dict, data, cfg: dict):
    """Milestone 3 Analysis 1: t-SNE, misclassified overlay, confusion matrices, per-class F1 on Roman-empire."""
    viz_models = {k: v for k, v in model_records.items() if k != "random"}
    n = len(viz_models)
    if n == 0:
        return

    tsne_cfg = cfg.get("visualization", {}).get("tsne", {})
    perplexity = int(tsne_cfg.get("perplexity", 30))
    max_iter = int(tsne_cfg.get("max_iter", tsne_cfg.get("n_iter", 1000)))
    random_state = int(tsne_cfg.get("random_state", 42))

    sample_size = min(4000, data.num_nodes)
    rng = np.random.default_rng(42)
    sample = rng.choice(data.num_nodes, sample_size, replace=False)

    embeddings_2d = {}
    metrics_map = {}
    for model_name, model in viz_models.items():
        model.eval()
        emb = model.get_embeddings(data.x, data.edge_index).detach().cpu().numpy()
        embeddings_2d[model_name] = make_tsne(
            perplexity=perplexity, max_iter=max_iter, random_state=random_state
        ).fit_transform(emb[sample])
        metrics_map[model_name] = compute_all_metrics(model, data)

    y_sample = data.y[sample].cpu().numpy()
    test_mask_full = data.test_mask.cpu().numpy()
    test_mask_sample = test_mask_full[sample]
    y_full = data.y.cpu().numpy()

    # 1) Ground-truth colored t-SNE
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for axis, (model_name, proj) in zip(axes, embeddings_2d.items()):
        m = metrics_map[model_name]
        axis.scatter(proj[:, 0], proj[:, 1], c=y_sample, s=6, cmap="tab20", alpha=0.7)
        axis.set_title(
            f"{MODEL_LABELS[model_name]}\nAcc: {m['accuracy']:.3f} | F1: {m['macro_f1']:.3f}",
            fontweight="bold",
        )
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Roman-empire Learned Representations (t-SNE, sampled 4,000 nodes)", fontweight="bold")
    fig.savefig(PLOTS / "roman_embedding_tsne.png", dpi=180)
    plt.close(fig)

    # 2) Misclassification overlay t-SNE (Milestone 3 Slide 7 example)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]
    test_indices_full = np.where(test_mask_full)[0]
    for axis, (model_name, proj) in zip(axes, embeddings_2d.items()):
        m = metrics_map[model_name]
        full_preds = np.full(data.num_nodes, -1)
        full_preds[test_indices_full] = m["predictions"]
        sample_preds = full_preds[sample]

        correct_s = (sample_preds == y_sample) & test_mask_sample
        wrong_s = (sample_preds != y_sample) & test_mask_sample
        non_test_s = ~test_mask_sample

        axis.scatter(proj[non_test_s, 0], proj[non_test_s, 1], c="lightgray", s=3, alpha=0.25)
        axis.scatter(proj[correct_s, 0], proj[correct_s, 1], c=y_sample[correct_s], cmap="tab20", s=8, alpha=0.65)
        axis.scatter(
            proj[wrong_s, 0],
            proj[wrong_s, 1],
            c="red",
            marker="x",
            s=18,
            linewidths=0.8,
            alpha=0.85,
            label=f"Errors ({wrong_s.sum()}/{test_mask_sample.sum()})",
        )
        axis.set_title(f"{MODEL_LABELS[model_name]} (Acc: {m['accuracy']:.3f})", fontweight="bold")
        axis.legend(fontsize=8, loc="upper right")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Roman-empire t-SNE with Misclassified Test Nodes (Red ×)", fontweight="bold")
    fig.savefig(PLOTS / "roman_tsne_misclassified.png", dpi=180)
    plt.close(fig)

    # 3) Normalized Confusion Matrices
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.8), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for axis, (model_name, m) in zip(axes, metrics_map.items()):
        cm = m["confusion_matrix"].astype(float)
        row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
        cm_norm = cm / row_sums
        sns.heatmap(cm_norm, ax=axis, cmap="YlOrRd", vmin=0, vmax=1, cbar=(axis == axes[-1]))
        axis.set_title(MODEL_LABELS[model_name], fontweight="bold")
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
    fig.suptitle("Normalized Confusion Matrices — Roman-empire (18 classes)", fontweight="bold")
    fig.savefig(PLOTS / "roman_confusion_matrices.png", dpi=180)
    plt.close(fig)

    # 4) Per-Class F1 comparison
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    x = np.arange(data.num_classes)
    width = 0.8 / n
    for idx, (model_name, m) in enumerate(metrics_map.items()):
        label = MODEL_LABELS[model_name]
        ax.bar(
            x + idx * width,
            m["per_class_f1"],
            width=width,
            color=PLOT_COLORS[label],
            label=label,
            alpha=0.88,
            edgecolor="white",
        )
    ax.set_xlabel("Class ID")
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-Class F1 Scores — Roman-empire", fontweight="bold")
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels([f"C{c}" for c in range(data.num_classes)], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    fig.savefig(PLOTS / "roman_per_class_f1.png", dpi=180)
    plt.close(fig)


def save_robustness(model_records: dict, data, device: torch.device, cfg: dict, quick: bool, roman_params: dict):
    """Milestone 3 Analysis 2: Edge dropout, feature noise, and label scarcity on Roman-empire."""
    robust_models = {k: v for k, v in model_records.items() if k != "random"}
    rob_cfg = cfg.get("robustness", {})
    edge_rates = rob_cfg.get("edge_drop_rates", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    noise_stds = rob_cfg.get("feature_noise_stds", [0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
    label_fracs = [0.1, 1.0] if quick else rob_cfg.get("label_fractions", [0.05, 0.1, 0.2, 0.4, 1.0])
    num_runs = 2 if quick else int(rob_cfg.get("num_perturbation_runs", 5))

    rows = []
    for model_name, model in robust_models.items():
        for rate in edge_rates:
            mean, std = evaluate_with_edge_dropout(model, data, float(rate), device, num_runs=num_runs)
            rows.append({
                "model": MODEL_LABELS[model_name],
                "condition": "edge_dropout",
                "level": float(rate),
                "accuracy": float(mean),
                "std": float(std),
            })
        for noise in noise_stds:
            mean, std = evaluate_with_feature_noise(model, data, float(noise), device, num_runs=num_runs)
            rows.append({
                "model": MODEL_LABELS[model_name],
                "condition": "feature_noise",
                "level": float(noise),
                "accuracy": float(mean),
                "std": float(std),
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES / "roman_robustness.csv", index=False)

    scarcity_rows = []
    scarcity_epochs = 30 if quick else 300
    scarcity_patience = 8 if quick else 30
    for model_name in robust_models:
        selected = roman_params.get(model_name, {})
        kwargs = model_kwargs(model_name, cfg, quick, selected)
        lr = float(selected.get("lr", cfg["models"][model_name].get("lr", 0.005)))
        wd = float(selected.get("weight_decay", cfg["models"][model_name].get("weight_decay", 5e-4)))
        for frac in label_fracs:
            acc, f1 = evaluate_with_label_scarcity(
                model_name,
                data,
                device,
                train_fraction=float(frac),
                seed=42,
                max_epochs=scarcity_epochs,
                patience=scarcity_patience,
                lr=lr,
                weight_decay=wd,
                **kwargs,
            )
            scarcity_rows.append({
                "model": MODEL_LABELS[model_name],
                "label_fraction": float(frac),
                "accuracy": float(acc),
                "macro_f1": float(f1),
            })

    scarcity_df = pd.DataFrame(scarcity_rows)
    scarcity_df.to_csv(TABLES / "roman_label_scarcity.csv", index=False)

    # Combined robustness degradation summary table
    summary_rows = []
    for model_name in robust_models:
        label = MODEL_LABELS[model_name]
        e_sub = frame[(frame.model == label) & (frame.condition == "edge_dropout")].sort_values("level")
        n_sub = frame[(frame.model == label) & (frame.condition == "feature_noise")].sort_values("level")
        l_sub = scarcity_df[scarcity_df.model == label].sort_values("label_fraction")

        clean_acc = float(e_sub["accuracy"].iloc[0])
        edge_drop_delta = clean_acc - float(e_sub["accuracy"].iloc[-1])
        noise_drop_delta = float(n_sub["accuracy"].iloc[0]) - float(n_sub["accuracy"].iloc[-1])
        label_drop_delta = float(l_sub["accuracy"].iloc[-1]) - float(l_sub["accuracy"].iloc[0])
        summary_rows.append({
            "model": label,
            "clean_accuracy": clean_acc,
            "edge_drop_50_acc": float(e_sub["accuracy"].iloc[-1]),
            "edge_degradation": edge_drop_delta,
            "noise_1_0_acc": float(n_sub["accuracy"].iloc[-1]),
            "noise_degradation": noise_drop_delta,
            "min_label_acc": float(l_sub["accuracy"].iloc[0]),
            "label_degradation": label_drop_delta,
        })
    pd.DataFrame(summary_rows).to_csv(TABLES / "robustness_summary.csv", index=False)

    # 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for condition, axis, title, xlabel in [
        ("edge_dropout", axes[0], "Edge Dropout", "Fraction of undirected edges removed"),
        ("feature_noise", axes[1], "Feature Noise", "Gaussian noise std (σ)"),
    ]:
        subset = frame[frame.condition == condition]
        for model_name in robust_models:
            label = MODEL_LABELS[model_name]
            group = subset[subset.model == label].sort_values("level")
            axis.errorbar(
                group["level"],
                group["accuracy"],
                yerr=group["std"],
                marker="o",
                capsize=3,
                linewidth=2,
                color=PLOT_COLORS[label],
                label=label,
            )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Test Accuracy")
        axis.set_ylim(0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    ax3 = axes[2]
    for model_name in robust_models:
        label = MODEL_LABELS[model_name]
        group = scarcity_df[scarcity_df.model == label].sort_values("label_fraction")
        ax3.plot(
            group["label_fraction"] * 100,
            group["accuracy"],
            marker="D",
            linewidth=2,
            color=PLOT_COLORS[label],
            label=label,
        )
    ax3.set_title("Label Scarcity", fontweight="bold")
    ax3.set_xlabel("Training Labels Used (%)")
    ax3.set_ylabel("Test Accuracy")
    ax3.set_ylim(0, 1.0)
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize=8)

    fig.suptitle("Robustness Analysis on Roman-empire", fontweight="bold")
    fig.savefig(PLOTS / "roman_robustness.png", dpi=180)
    plt.close(fig)


def write_latex(summary: pd.DataFrame, quick: bool):
    ordered = summary.copy()
    ordered["acc_fmt"] = ordered.apply(lambda r: f"{r['accuracy']:.3f} $\\pm$ {r['accuracy_std']:.3f}", axis=1)
    ordered["f1_fmt"] = ordered.apply(lambda r: f"{r['macro_f1']:.3f} $\\pm$ {r['macro_f1_std']:.3f}", axis=1)
    table = ordered.sort_values(["dataset", "model"])[["dataset", "model", "acc_fmt", "f1_fmt"]]
    rows = "\n".join(" & ".join(map(str, row)) + " \\\\" for row in table.itertuples(index=False, name=None))
    mode_note = "Quick pipeline validation run (1 seed, 30 epochs)." if quick else "Full 5-seed benchmark with validation-selected hyperparameters."

    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\title{Across the Homophily Spectrum: A Controlled Study of GNNs on Node Classification}
\author{Sushmit Biswas, Anurag Rao, Yashwant Reddy, Bhavesh Mallepeddi}
\date{November 2026}
\begin{document}
\maketitle
\begin{abstract}
This report studies node classification across Amazon Computers, WikiCS, and Roman-empire. The datasets move from high-homophily co-purchase and medium-homophily hyperlink graphs to a strongly heterophilic word-relationship graph. GATv2 (dynamic attention), GraphSAGE (mean neighborhood aggregation with ego skip connection), and H2GCN (ego/neighbor separation and 2-hop concatenation) are evaluated alongside MLP (feature-only) and Random (floor) baselines. All models use identical task definitions and validation-selected checkpoints.
\end{abstract}
\section{Research Question}
Do GNN inductive biases transfer across graphs with different homophily, or does the best architecture depend on whether neighboring labels agree? In particular, does an architecture explicitly designed for heterophily (H2GCN) or ego-separated aggregation (GraphSAGE) outperform attention-based message passing (GATv2) on low-homophily graphs?
\section{Datasets and Task}
We perform semi-supervised node classification. Amazon Computers nodes are products and edges indicate co-purchase relationships (60/20/20 stratified split). WikiCS nodes are computer-science articles and edges represent hyperlinks (official split 0). Roman-empire nodes are words and edges encode word adjacency and syntactic dependency links (official split 0).
\section{Method}
Hyperparameters (learning rate, weight decay, dropout, and hidden width) are selected per model-dataset pair using validation accuracy only; the test split is never used for model selection. """ + mode_note + r"""
\section{Results}
\begin{table}[h]
\centering
\caption{Validation-selected test results (mean $\pm$ std) for every model-dataset pair.}
\begin{tabular}{llll}
\toprule
Dataset & Model & Accuracy & Macro-F1 \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
\begin{figure}[h]
\centering
\includegraphics[width=0.95\linewidth]{results/plots/benchmark_comparison.png}
\caption{Accuracy and Macro-F1 across all model-dataset pairs.}
\end{figure}
\begin{figure}[h]
\centering
\includegraphics[width=0.95\linewidth]{results/plots/training_curves.png}
\caption{Training and validation accuracy curves (seed 42).}
\end{figure}
\section{Roman-empire Analyses}
Roman-empire is the largest dataset by node count (22,662 nodes). We run two analyses across the trainable models (GATv2, GraphSAGE, H2GCN, MLP): representation visualization (t-SNE, misclassification overlay, confusion matrices, and per-class F1) and robustness (undirected edge dropout, Gaussian feature noise, and training label scarcity).
\begin{figure}[h]
\centering
\includegraphics[width=0.95\linewidth]{results/plots/roman_embedding_tsne.png}
\caption{Sampled t-SNE view of penultimate representations on Roman-empire.}
\end{figure}
\begin{figure}[h]
\centering
\includegraphics[width=0.95\linewidth]{results/plots/roman_tsne_misclassified.png}
\caption{t-SNE with misclassified test nodes highlighted in red on Roman-empire.}
\end{figure}
\begin{figure}[h]
\centering
\includegraphics[width=0.95\linewidth]{results/plots/roman_robustness.png}
\caption{Robustness curves (edge dropout, feature noise, label scarcity) on Roman-empire.}
\end{figure}
\section{Conclusion}
The project tests a structural hypothesis across the homophily spectrum. Interpreting accuracy alongside the MLP baseline, t-SNE structure, and perturbation curves clarifies when graph edges help versus hurt node classification.
\end{document}
"""
    (ROOT / "report.tex").write_text(tex, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run Project-2026 GNN Benchmark & Milestone 3 Analyses")
    parser.add_argument("--quick", action="store_true", help="Fast 1-seed validation run (safe on CPU).")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip grid search and use defaults in hyperparams.yaml.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Override random seeds.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None, help="Compute device.")
    args = parser.parse_args()

    cfg = load_config()
    seeds = args.seeds if args.seeds else ([42] if args.quick else cfg["global"]["seeds"])
    device = resolve_device(args.device, cfg)

    for directory in [TABLES, PLOTS, MODELS]:
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    tuning_rows = []
    best_params = {}
    first_seed_histories = {}
    roman_models = {}
    roman_data = None
    roman_params = {}

    for dataset_name in DATASET_ORDER:
        data = load_dataset(dataset_name, root=str(ROOT / "data"))
        best_params[dataset_name] = {}
        first_seed_histories[data.dataset_name] = {}
        print(f"\n=== Dataset: {data.dataset_name} ({data.num_nodes} nodes, {data.num_edges} edges) ===")

        for model_name in get_all_model_names():
            selected, trials = tune_pair(model_name, data, device, cfg, args.quick, skip_tuning=args.skip_tuning)
            best_params[dataset_name][model_name] = selected
            tuning_rows.extend(trials)
            if dataset_name == "roman-empire":
                roman_params[model_name] = selected

            for seed in seeds:
                result, model, data_device, history = train_one(
                    model_name, data, device, cfg, seed, args.quick, selected
                )
                rows.append(result)
                if seed == seeds[0] and history is not None:
                    first_seed_histories[data.dataset_name][model_name] = history
                if dataset_name == "roman-empire" and seed == seeds[0]:
                    roman_models[model_name] = model
                    roman_data = data_device
                print(
                    f"  {result['dataset']} | {result['model']:10s} | seed={seed} | "
                    f"acc={result['accuracy']:.4f} | f1={result['macro_f1']:.4f} | {result['seconds']:.1f}s"
                )

    raw = pd.DataFrame(rows)
    raw.to_csv(TABLES / "benchmark_per_seed.csv", index=False)

    summary = (
        raw.groupby(["dataset", "model"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            best_val_accuracy=("best_val_accuracy", "mean"),
            seconds=("seconds", "mean"),
            parameters=("parameters", "first"),
        )
        .fillna(0.0)
    )
    summary.to_csv(TABLES / "benchmark_summary.csv", index=False)

    if tuning_rows:
        pd.DataFrame(tuning_rows).to_csv(TABLES / "tuning_results.csv", index=False)
    (RESULTS / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    save_benchmark_plot(summary)
    save_training_curves(first_seed_histories)

    if roman_data is not None:
        save_visualisation(roman_models, roman_data, cfg)
        save_robustness(roman_models, roman_data, device, cfg, args.quick, roman_params)

    manifest = {
        "quick": args.quick,
        "skip_tuning": args.skip_tuning,
        "seeds": seeds,
        "device": str(device),
        "datasets": DATASET_ORDER,
        "models": get_all_model_names(),
        "validation_tuning": best_params,
    }
    (TABLES / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_latex(summary, args.quick)
    print(f"\nDone! Saved {len(raw)} runs, plots, CSV tables, and report.tex.")


if __name__ == "__main__":
    main()
