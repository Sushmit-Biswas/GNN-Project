# Master Presentation Flow 
*(Milestone 1: Proposal & Motivation)*

---

## PART 1: Introduction & The Homophily Spectrum 
*(Presented by Sushmit)*

### Slide 1: Across the Homophily Spectrum
**Subtitle:** Can Standard GNNs Generalize When "Opposites Attract"?
- **Team Members:** Sushmit Biswas, Yashwant Reddy, Anurag Rao, Bhavesh Mallepeddi
- **Core Hypothesis:** GNNs rely on neighborhood homophily—how well do GATv2, GraphSAGE, and H2GCN adapt when neighbor labels actively diverge?

### Slide 2: The Task Formulation
**Title:** Semi-Supervised Node Classification
*(Visual Recommendation: `classification.png`)*
- **Given:** A partially labeled graph G = (V, E, X) with sparse training seeds (5-10% of nodes).
- **Goal:** Predict the specific class labels for all remaining unobserved nodes.
- **Why Semi-Supervised?** Tests label efficiency; our GNNs must learn patterns that propagate from a tiny subset of known nodes to the vast unlabeled graph.

### Slide 3: Dataset Characteristics across the Spectrum
**Title:** Introducing our Three Datasets
*(Visual Recommendation: `spectrum.png`)*
- We selected three datasets representing different regions of the Homophily Spectrum:
1. **Amazon Computers:** High Homophily (h ≈ 0.78) - Neighbors likely share labels.
2. **WikiCS:** Medium Homophily (h ≈ 0.65) - Links are useful but imperfect.
3. **Roman-empire:** Extreme Heterophily (h ≈ 0.05) - Neighbors actively differ in labels.

---

## PART 2: Amazon Computers Deep-Dive
*(Presented by Sushmit)*

### Slide 4: Why Amazon Computers?
**Title:** A High-Homophily Co-Purchase Network
- Amazon Computers is a real-world graph dataset containing individual computer-related products scraped from Amazon.
- The products are connected via "frequently bought together" checkout links.
- The target classes are distinct Amazon product categories (e.g., Laptops, Desktops, Cables).
- This dataset guarantees testing on an environment where homophily is extremely high—meaning neighbor signals heavily reinforce the class.

### Slide 5: Amazon Computers Dataset Specifications
**Title:** What Is in the Dataset?
*(Visual Recommendation: `gen_amazon_node.png`)*
- **Nodes:** 13,752 computer products 
- **Edges:** 491,722 co-purchase links 
- **Node features:** 767-dimensional Bag-of-Words (BoW) from product reviews 
- **Classes:** 10 product categories 
- **Edge homophily:** ~0.78 (Very High Homophily) 

### Slide 6: What Are the Nodes and Edges?
**Title:** Meaning of the Graph Elements
*(Visual Recommendation: `gen_amazon_graph.png`)*
- **Node:** A single distinct product on Amazon.
- **Connection:** Two products are connected if Amazon records them as "frequently bought together".
- **Meaning of a link:** High likelihood of complementary use.
- **Graph Type:** Undirected. If user buys Mouse with Keyboard, the relationship is mutual.

### Slide 7: What Are the Node Features?
**Title:** Product Reviews as Numerical Input
- Every node has a **767-dimensional feature vector**.
- It is created using a **Bag-of-Words (BoW)** encoding of the product's verified reviews.
- The feature matrix has shape $X \in \mathbb{R}^{13752 \times 767}$.
- Each column represents the presence or frequency of a specific vocabulary word found across the dataset's reviews.

### Slide 8: Combining Content & Context
**Title:** Why do we need both Features and Edges?
*(Visual Recommendation: `mouse.png`)*
For this dataset, we see **High Homophily (0.78)**. People buy similar product categories together.
- **Feature Signal (Content):** Node features might tell us a product is a "mouse" based on words in the reviews. But they might not be able to tell us exactly *what kind* of mouse. 
- **Structural Signal (Context):** A node is co-purchased with an **RTX 4090 GPU**, a **Mechanical Keyboard**, and a **Gaming Chair**. 
- **The Complement (Synergy):** The features tell us it's a mouse, but the edges explicitly tell us it's a **Gaming Mouse**. Because homophily is very high, the structure strongly refines and confirms the feature signal!

*(Sushmit hands off to Teammate 1)*: *"Now that we've seen how GNNs perform when Homophily is high, I'll pass it to [Friend 1] to walk you through WikiCS, our Medium Homophily dataset."*

---

## PART 3: WikiCS Deep-Dive
*(Presented by Teammate 1)*

*(Note: Teammate 1 will present their 7 WikiCS slides here. See `GNN - WikiCS.pdf`. They should use `gen_wikics_graph.png` and `gen_wikics_node.png` if they need graphics!)*

---

## PART 4: Roman-empire Deep-Dive
*(Presented by Teammate 2 & 3)*

### Slide 9: Why Roman-empire?
**Title:** Extreme Heterophily in Syntactic Dependencies
- Roman-empire is a linguistically-focused, real-world graph dataset.
- It contains individual words parsed from the "Roman Empire" Wikipedia article.
- The target classes are 18 distinct grammatical roles.
- This dataset is critical for our hypothesis: what happens to GNNs when homophily is essentially non-existent (h=0.05). 

### Slide 10: Roman-empire Dataset Specifications
**Title:** What Is in the Dataset?
*(Visual Recommendation: `gen_roman_node.png`)*
- **Nodes:** 22,662 English words 
- **Edges:** 65,856 syntactic links 
- **Node features:** 300-dimensional FastText word embeddings 
- **Classes:** 18 syntactic roles 
- **Edge homophily:** ~0.05 (Extreme Heterophily) 

### Slide 11: What Are the Nodes and Edges?
**Title:** Meaning of the Graph Elements
*(Visual Recommendation: `roman_empire_syntax.png` or `gen_roman_graph.png`)*
- **Node:** A single word within the text.
- **Connection:** Syntactic dependency parse tree links (e.g., an adjective modifying a noun).
- **Meaning of a link:** "Opposites attract." By the laws of grammar, words rarely connect to other words of the exact same grammatical role.

### Slide 12: What Are the Node Features?
**Title:** FastText Embeddings as Numerical Input
- Every node has a **300-dimensional feature vector**.
- It is generated using **FastText** embeddings.
- Unlike Bag of Words, FastText captures sub-word morphology (prefixes, suffixes), heavily informing the grammatical class (e.g., words ending in "-ly").

### Slide 13: The Heterophily Anchor
**Title:** Using Features to Save Graph Structure
*(Visual Recommendation: `heterophily_anchor.png` or `gen_anchor.png`)*
- **Feature Signal (Content):** Dictionary meaning and sub-word structure of the text.
- **Structural Signal (Context):** Syntactic grammar. 
- **The Complement (The Anchor):** Because edge homophily is 0.05, neighbor messages consist entirely of different classes. Standard GNNs (GATv2, GraphSAGE) will over-smooth and mathematically mix adjectives into nouns if they only use structure, causing catastrophic failure. Here, the node features act as a semantic **anchor**, grounding the prediction in the word's original identity and preventing complete structural corruption!
