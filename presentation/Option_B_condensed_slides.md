# Amazon Computers & Roman-empire Presentation Slides

*Note: Since your teammate gave a deep-dive definition of GNN concepts (like what a feature is, what an edge is) for WikiCS, you can use these 4 highly focused slides for your remaining datasets to hit all the rubric requirements without boring the audience or repeating definitions.*

---

## Dataset 2: Amazon Computers

### Slide 1: Amazon Computers - Specs & Semantics

**Title:** A High-Homophily Co-Purchase Network

| Property | Amazon Computers |
|---:|:---|
| **Nodes** | 13,752 (Individual Computer Products on Amazon) |
| **Edges** | 491,722 ("Frequently Bought Together" links) |
| **Node Features** | 767-dimensional Bag-of-Words (BoW) from verified customer reviews |
| **Classes** | 10 (Product Categories, e.g., laptops, cables, peripherals) |
| **Edge Homophily** | ~0.78 (High Homophily) |
| **Task** | Semi-Supervised Node Classification (Predict product category) |

**Interpretation:** Each node is a product. An edge exists if two products are frequently bought together. The features represent the text reviewers used to describe the product.

**Visual recommendation:** An arrow pointing from a "Gaming Laptop" node to a "Gaming Mouse" node, with an Amazon logo or shopping cart icon. 

---

### Slide 2: Amazon Computers - Combining Content & Context

**Title:** Why do we need both Features and Edges?

*(Show `mouse.png` on this slide)*

For this dataset, we see **High Homophily (0.78)**. People buy similar product categories together.

- **Feature Signal (Content):** Node features might tell us a product is a "mouse" based on words in the reviews. But they might not be able to tell us exactly *what kind* of mouse. 
- **Structural Signal (Context):** A node is co-purchased with an **RTX 4090 GPU**, a **Mechanical Keyboard**, and a **Gaming Chair**. 
- **The Complement (Synergy):** The features tell us it's a mouse, but the edges explicitly tell us it's a **Gaming Mouse** (as opposed to an Office Mouse that is co-purchased with a Desk Lamp and USB Hub). Because homophily is very high, the structure strongly refines and confirms the feature signal!

---

## Dataset 3: Roman-empire

### Slide 3: Roman-empire - Specs & Semantics

**Title:** Extreme Heterophily in Syntactic Dependencies

| Property | Roman-empire |
|---:|:---|
| **Nodes** | 22,662 (Individual words from the Roman Empire Wikipedia article) |
| **Edges** | 65,856 (Syntactic dependency links in a parse tree) |
| **Node Features** | 300-dimensional FastText word embeddings |
| **Classes** | 18 (Syntactic roles in the sentence: verb, noun, subject, etc.) |
| **Edge Homophily** | ~0.05 (Very Low / Extreme Heterophily) |
| **Task** | Semi-Supervised Node Classification (Predict grammatical role) |

**Interpretation:** Rather than articles or products, the nodes here are individual words. The edges are grammatical links (e.g., an adjective modifying a noun). 

**Visual recommendation:** A sentence tree diagram. e.g., "The [Article] -> quick [Adjective] -> fox [Noun]", where the words are nodes and the arrows are dependency edges.

---

### Slide 4: Roman-empire - The "Opposites Attract" Problem

**Title:** The Heterophily Anchor

For this dataset, we see **Extreme Heterophily (0.05)**. This is a massive challenge for standard GNNs. 

- **Feature Signal (Content):** FastText embeddings provide the standalone semantic meaning of the word itself. 
- **Structural Signal (Context):** Syntactic links show grammatical relationships. By definition of grammar, an adjective usually links to a noun, or a subject links to a verb. They rarely link to the same class!
- **The Complement (The Anchor):** If we only use structural edges, a standard GNN (like GraphSAGE) will mix the features of adjectives with nouns, effectively destroying the signal ("over-smoothing"). Here, **Node Features are absolutely critical**. They act as a semantic anchor that prevents the structure from completely wiping out the word's original identity when "opposites attract".

**Visual recommendation:** A balance scale or visual tug-of-war. Show standard aggregation mixing different colors into a muddy brown, while the original "Node Feature" acts as a bright anchor preserving the original color.

---
