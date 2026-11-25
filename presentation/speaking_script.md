# 🎤 Sushmit's Speaking Script — Milestone 1
*(Don't memorize this. Just read it a few times so you know what to say on each slide.)*

---

## Slide 1: Cover — "Across the Homophily Spectrum"

> "Hi everyone. Our project is called **Across the Homophily Spectrum**. The main question we are trying to answer is — can standard GNNs still work well when the connected nodes in a graph are very different from each other? In other words, what happens when opposites attract?"

---

## Slide 2: Team

> "Our team has four members: Sushmit Biswas, Anurag Rao, Yashwant Reddy, and Bhavesh Mallepeddi."

*(Just read the names and move to the next slide.)*

---

## Slide 3: The Task Formulation

> "So what is the task we are solving? It is **Semi-Supervised Node Classification**. We are given a graph, but only a small number of nodes — around 5 to 10 percent — have labels. The rest are unknown. Our goal is to predict the correct labels for all the unknown nodes."

> "Why semi-supervised? Because in real life, labeling every node is very expensive. So we want to check if our GNN models can learn from just a few labeled nodes and use that to classify the rest."

*(Point to the colored nodes in the graphic and say "these are the known ones", then point to the grey question-mark nodes and say "and these are the ones we need to predict".)*

---

## Slide 4: Dataset Characteristics across the Spectrum

> "Now let me introduce our three datasets. We picked these three because they cover different levels of homophily."

> "On the left side we have **Amazon Computers** — this has high homophily, which means connected nodes usually have the same label. In the middle we have **WikiCS** — medium homophily, so the links help but they are not always correct. And on the right we have **Roman-empire** — this has very low homophily, meaning connected nodes almost always have different labels."

> "This spread lets us test whether standard GNN models break down when homophily drops, and whether a model like H2GCN, which is built for this problem, can handle it better."

*(Point from left to right across the spectrum graphic as you talk.)*

---

## Slide 5: Why Amazon Computers?

> "Now I will talk about my dataset in detail: **Amazon Computers**."

> "This is a real-world graph dataset where each node is a computer product on Amazon. The edges between them are co-purchase links — meaning if customers often buy two products together, those products are connected. And our job is to predict which of 10 product categories each product belongs to — like Laptops, Desktops, or Cables."

---

## Slide 6: What Is in the Dataset?

> "Here are the numbers. The dataset has **13,752 nodes** — so 13,752 products. These are connected by about **4.9 lakh co-purchase edges**. Each product has a **767-dimensional feature vector**, which is a Bag-of-Words built from real customer reviews. There are **10 classes** to predict, and the homophily is about **0.78** — so 78 percent of the time, two connected products belong to the same category."

*(Just read through the table. Keep it quick.)*

---

## Slide 7: Nodes, Edges, & Features

> "Let me explain what the graph elements actually mean here."

> "On the left side — the **graph structure**. Each node is a product. The edges are undirected links that mean 'frequently bought together'. So if someone buys a mouse with a keyboard, both products get connected."

> "On the right side — the **node features**. Each product has a 767-dimensional Bag-of-Words vector. This vector is built from the actual reviews that real customers wrote about that product. Each number in the vector tells us how often a specific word appears in those reviews."

*(Point to the left column of the slide when talking about structure, and the right column when talking about features.)*

---

## Slide 8: Why Do We Need Both Features and Edges?

> "This is the most important slide in my section."

*(Point to the left side of the mouse.png graphic.)*

> "Look at this example. Say we have a product and the review text tells us it is a 'mouse'. But that alone is not enough — what kind of mouse is it? Now look at its co-purchase links. This mouse is bought together with an **RTX 4090 GPU**, a **Mechanical Keyboard**, and a **Gaming Chair**. So the edges are clearly telling us — this is a **Gaming Mouse**."

*(Now point to the right side of the graphic.)*

> "But what if the same 'mouse' is instead bought with a **Webcam**, a **USB Hub**, and a **Desk Lamp**? Then the edges tell us it is an **Office Mouse**."

> "So the bottom line is simple — **features tell us it's a mouse, but edges tell us what KIND of mouse**. And because homophily is high in this dataset, the edges give us the exact missing clues that the features could not provide."

*(Pause for a moment.)*

---

## Handoff

> "So that was Amazon Computers — our high-homophily dataset. Now I will hand it over to [Teammate's Name], who will talk about WikiCS, our medium-homophily dataset."

*(Step back and let your teammate take over.)*

---

### 🕐 Total Speaking Time: ~4 minutes
