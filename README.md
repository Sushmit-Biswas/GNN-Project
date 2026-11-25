# 🌐 GNN Project: "Across the Homophily Spectrum"

## Node Classification on Graphs with Varying Structural Assumptions

### IIIT-B — Semester 5 | Graph Neural Networks

Team: **Sushmit Biswas**, **Anurag Rao**, **Yashwant Reddy**, **Bhavesh Mallepeddi**

---

## 📋 Project Overview

This project investigates how **Graph Neural Network architectures** perform across datasets spanning the **homophily spectrum** — from high homophily (neighbors share labels) to extreme heterophily (neighbors differ).

> _Hypothesis: Standard GNNs (like GATv2) excel on homophilic graphs but degrade on heterophilic graphs, where architectures with explicit ego-neighbor separation and higher-order aggregation (H2GCN, GraphSAGE) outperform them._

---

## 🚀 How to Run (One Codebase for NVIDIA GPU, Google Colab, and CPU)

The single entry point **`run_project.py`** works seamlessly on **NVIDIA GPUs (e.g., RTX 5090)**, **Google Colab (T4/A100)**, and **local CPU**.

### Option A: Google Colab (Recommended if no local NVIDIA GPU)
1. Open [notebooks/00_run_on_colab.ipynb](notebooks/00_run_on_colab.ipynb) in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run all cells — it clones the repo, installs PyG, runs the full 5-seed benchmark + tuning + Milestone 3 plots (~10–15 min), and zips `results/` + `report.tex` for download.

### Option B: Local / Friend's NVIDIA GPU (e.g., RTX 5090)
```bash
git clone https://github.com/Sushmit-Biswas/GNN-Project.git
cd GNN-Project
pip install -r requirements.txt
pytest -q test_pipeline.py
python run_project.py --device cuda
```
*(Runs all tuning, 5-seed training, t-SNE, confusion matrices, and robustness analyses in ~2–5 minutes on an RTX 5090.)*

### Option C: Local CPU Quick Sanity Check
```bash
python run_project.py --quick --device cpu
```
*(Runs 1 seed and 30 epochs only to validate the pipeline; do not use `--quick` numbers for the final presentation.)*

---

## 📁 Generated Outputs

- `results/tables/benchmark_per_seed.csv` — Per-seed accuracy, Macro-F1, runtime, selected hyperparameters
- `results/tables/benchmark_summary.csv` — Mean ± std across all 5 seeds
- `results/tables/tuning_results.csv` & `results/best_params.json` — Full validation tuning log
- `results/tables/roman_robustness.csv`, `roman_label_scarcity.csv`, `robustness_summary.csv`
- `results/tables/experiment_manifest.json` — Exact run metadata (`quick`, `seeds`, `device`)
- `results/plots/benchmark_comparison.png` — Bar charts with error bars
- `results/plots/training_curves.png` — Train/val accuracy curves across datasets & models
- `results/plots/roman_embedding_tsne.png` & `roman_tsne_misclassified.png` — t-SNE embeddings + error overlay
- `results/plots/roman_confusion_matrices.png` & `roman_per_class_f1.png`
- `results/plots/roman_robustness.png` — Edge dropout, feature noise, and label scarcity curves
- `results/models/<dataset>/<model>/seed_<seed>.pt` — Validation-selected checkpoints
- `report.tex` — Auto-generated LaTeX summary report

---

## 📊 Datasets & Split Protocols

| Dataset              | Domain                 | Nodes  | Edges   | Features | Classes | Edge Homophily | Split Protocol |
| -------------------- | ---------------------- | ------ | ------- | -------- | ------- | -------------- | -------------- |
| **Amazon Computers** | E-commerce co-purchase | 13,752 | 491,722 | 767      | 10      | ~0.78 (High)   | 60/20/20 stratified (`random_state=42`) |
| **WikiCS**           | Wikipedia CS articles  | 11,701 | 431,726 | 300      | 10      | ~0.65 (Medium) | Official split 0 |
| **Roman-empire**     | Word dependency graph  | 22,662 | 65,854  | 300      | 18      | ~0.05 (Low)    | Official split 0 |

---

## 🧠 Architectures & Baselines

| Model         | Key Mechanism                                            | Homophily Bias                  | Paper                         |
| ------------- | -------------------------------------------------------- | ------------------------------- | ----------------------------- |
| **GATv2**     | Dynamic multi-head attention                             | Implicit (neighbor aggregation) | Brody et al., ICLR 2022       |
| **GraphSAGE** | Mean neighborhood aggregation + separate self weight     | Neutral                         | Hamilton et al., NeurIPS 2017 |
| **H2GCN**     | Ego/neighbor separation + 1-hop & 2-hop concatenation    | Explicitly heterophily-aware    | Zhu et al., NeurIPS 2020      |
| **MLP**       | Feature-only (ignores graph structure)                   | Feature-only baseline           | —                             |
| **Random**    | Uniform random prediction                                | Floor baseline                  | —                             |

---

## 📈 Milestones

- **Milestone 1 (Proposal & Motivation):** 3 datasets across the homophily spectrum, node/edge/feature semantics, node classification formulation.
- **Milestone 2 (Methodology & Empirical Comparison):** 3 GNNs + 2 baselines, validation-only hyperparameter tuning, 5-seed mean ± std benchmarking (Accuracy & Macro-F1).
- **Milestone 3 (Insights & Analysis on Largest Dataset — Roman-empire):**
  1. **Visualization:** t-SNE penultimate embeddings, misclassified test-node overlay, confusion matrices, per-class F1.
  2. **Robustness:** Edge dropout (0–50%), Gaussian feature noise ($\sigma = 0.0\text{--}1.0$), label scarcity (5%–100%).
