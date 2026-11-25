# 🌐 GNN Project: "Across the Homophily Spectrum"
## Milestone 1: Proposal & Motivation

Topic: Across the Homophily Spectrum: Evaluating GNNs under Graph Disassortativity
Presentation Type: 5-Minute Technical Academic Pitch
Target Audience: Machine Learning / Computer Science Evaluators

---

### Slide 1: Across the Homophily Spectrum
**Subtitle:** Can Standard GNNs Generalize When "Opposites Attract"?

**Team Members:**
- Sushmit Biswas (Roll No: BT2024038)
- Yashwant Reddy (Roll No: BT2024103)
- Anurag Rao (Roll No: IMT2024028)
- Bhavesh Mallepeddi (Roll No: IMT2024069)

**Core Hypothesis:** "GNNs rely on neighborhood homophily—how well do GATv2, GraphSAGE, and H2GCN adapt when neighbor labels actively diverge?"

---

### Slide 2: Dataset Characteristics across the Spectrum

| Dataset | Homophily (h) | Nodes | Edges |
|---------|---------------|-------|-------|
| Amazon Computers | 🟢 0.78 (High) | 13.7K | 491K |
| WikiCS | 🟡 0.65 (Medium) | 11.7K | 216K |
| Roman-empire | 🔴 0.05 (Very Low) | 22.6K | 65.8K |

---

### Slide 3: Dataset Semantics (Nodes, Edges, Features)

**Amazon Computers:**
- **Nodes:** Individual products listed on Amazon.
- **Edges:** Frequent co-purchase link ("buy together").
- **Features:** Bag-of-words from verified customer reviews.

**WikiCS:**
- **Nodes:** Computer science articles.
- **Edges:** Outgoing hyperlinks.
- **Features:** GloVe word embeddings of the article text.

**Roman-empire:**
- **Nodes:** Individual words in a sentence.
- **Edges:** Syntactic dependency parse trees.
- **Features:** FastText word embeddings.

---

### Slide 4: Task Formulation: Node Classification

**Task: Semi-Supervised Node Classification**
Given a partially labeled graph G = (V, E, X) with sparse training seeds (5-10% of nodes), predict the specific class labels for all remaining unobserved nodes.

Classes to predict:
- **Amazon:** 10 product categories.
- **WikiCS:** 10 CS sub-fields.
- **Roman-empire:** 18 syntactic roles.

---

### Slide 5: Why are BOTH node features and structural edges necessary?

They provide orthogonal signals that fail when used in isolation:
- **Feature Signal Alone:** Product description specifies a "Computer Mouse" (generic). Needs context.
- **Topological Edge Alone:** Node is connected to RTX 4090 and Gaming Chair. Lacks identity.
- **Combined:** Disambiguates item explicitly as a "Gaming Mouse".

---

### Slide 6: How do the node features complement the graph structure?

Features plug the holes where pure topological structure fails:
1. **Handling Structural Sparsity:** When nodes are completely isolated (degree=0), features are the *only* way to classify them.
2. **The Heterophily Anchor:** In our Roman-empire dataset (h=0.05), neighboring edges connect entirely different classes. Standard structural aggregation destroys data here. Node features act as a semantic anchor preventing complete model failure when "opposites attract".

---

### Slide 7: Thank You & Next Steps
**Title:** Thank You / Q&A

- **Next Phase:** We will benchmark GATv2, GraphSAGE, and H2GCN.
- **Deliverable:** Accuracy curves vs. homophily levels to empirically validate the boundary conditions of standard GNNs.
- **Questions?**
