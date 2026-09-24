# 🛍️ Customer Segmentation using Unsupervised Learning

![Status](https://img.shields.io/badge/status-complete-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4-orange)

Segmenting ~1M e-commerce transactions into behavioral customer groups using RFM features and four clustering algorithms — with a focus on comparing *why* they disagree, not just running each one.

---

## 📑 Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Approach](#-approach)
- [Results at a Glance](#-results-at-a-glance)
- [Methodology & Findings](#-methodology--findings)
- [Key Finding](#-key-finding-customer-behavior-is-continuous-not-categorical)
- [Repo Structure](#-repo-structure)
- [How to Reproduce](#-how-to-reproduce)
- [Key Decisions Worth Noting](#-key-decisions-worth-noting)
- [Limitations](#-limitations)

---

## 🎯 Problem Statement

An online retailer has ~1M transactions but no systematic way to tell a loyal high-spender from a one-time buyer who's already gone. This project segments the customer base using unsupervised clustering (no labeled "customer type" exists to predict) — comparing four algorithms to identify high-value customers, churn risk, and outliers that need separate handling.


## 📦 Dataset

[Online Retail II](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) — ~1,067,371 transactions across two sheets (Dec 2009 – Dec 2011), UK-based online retailer.

> Not included in this repo (see [How to Reproduce](#-how-to-reproduce)) — download from the link above.

## 🧭 Approach

| Stage | What happened |
|---|---|
| **Cleaning** | Removed missing CustomerID, exact duplicates, £0 rows, cancelled orders |
| **Feature engineering** | Built RFM (Recency, Frequency, Monetary) per customer |
| **Preprocessing** | Log-transformed Monetary/Frequency for skew; scaled with StandardScaler |
| **Clustering** | Compared K-Means, Hierarchical, and DBSCAN on identical features |
| **Visualization** | PCA (linear) and UMAP (non-linear) projections, cross-checked against each other |

---

## 📊 Results at a Glance

**Four customer segments, from K-Means (k=4):**

| Segment | Recency | Frequency | Monetary | Description |
|---|---|---|---|---|
| 🏆 Champions | 43 days | 22.2 orders | £13,120 | Highly active, frequent, largest spenders |
| 💚 Loyal / Core | 99 days | 5.5 orders | £1,910 | Consistent, reliable engagement |
| 🌱 New / Low-Value | 106 days | 1.7 orders | £426 | Recently seen, rarely purchase |
| 🥀 Lost / Churned | 493 days | 1.7 orders | £538 | Inactive over a year |

**The bigger finding:** every technique used — PCA, elbow/silhouette, Hierarchical, DBSCAN, UMAP — independently pointed to the same conclusion: this customer base is a *spectrum*, not four clean boxes. Full reasoning below.

---

## 🔍 Methodology & Findings

### PCA
The projection shows one continuous, dense cloud — no visibly separated clusters — suggesting customer behavior exists on a spectrum rather than in distinct categories. Scattered points on the far right/top are the wholesale-pattern outliers flagged during EDA. 2 components retain **95.05%** of total variance, so this 2D view is a faithful stand-in for the full 3D feature space.


### K-Means
Selected **k=4** via elbow method + silhouette score (k=2 scored highest but was too coarse for business use). Produced 4 balanced clusters (965–1,891 customers each), but visibly imposed straight-line boundaries on data with no natural separation — a known K-Means limitation, later contrasted with DBSCAN's outlier handling.


### Hierarchical Clustering (Ward Linkage)
Cut to k=4 for direct comparison. Produced a less balanced split (595–2,329 customers) than K-Means. Cross-tabulating the two labelings showed strong agreement on 2 of 4 segments (96–100% overlap) — but K-Means' largest cluster was split across 3 different Hierarchical groups, evidence that a meaningful chunk of customers sit in a continuous, ambiguous region rather than a well-defined cluster.


### DBSCAN
`eps` was tuned by testing values from 0.20 to 0.33 — lower values fragmented the data into meaningless micro-clusters (4–11 customers) with high noise counts (400+); increasing `eps` progressively consolidated this into real structure. At **eps=0.33** (`min_samples=5`), DBSCAN converged on **3** substantial clusters (3,209 / 1,604 / 930) and flagged **135 customers (~2.3%) as noise** — customers K-Means and Hierarchical were structurally forced to absorb into a normal cluster. DBSCAN settling on 3 clusters, not 4, isn't a discrepancy — it's the finding: density-based structure here supports 3 natural groups plus a genuine outlier fringe.

### UMAP
Applied as a non-linear check on whether PCA's smooth cloud was hiding real structure. UMAP's projection looks strikingly different — several distinct blobs connected by thin bridges. But overlaying all three algorithms' cluster labels onto this same layout shows **none of them align with UMAP's visual blobs**. This confirms the separation is a projection artifact (driven by UMAP's local-neighborhood-preserving nature and `min_dist`), not evidence of genuine behavioral segments.

---

## 🧩 Key Finding: Customer Behavior Is Continuous, Not Categorical

Five independent pieces of evidence, from every technique in this project, point to the same conclusion:

1. **PCA** — raw projection shows one smooth cloud, no visible gaps.
2. **Elbow + Silhouette** — no sharp, unambiguous k; scores support k=2/3/4 roughly equally.
3. **K-Means vs. Hierarchical** — same k=4, but disagree substantially (largest K-Means cluster split 3 ways).
4. **DBSCAN** — given no target count, independently finds 3 clusters, not 4.
5. **UMAP** — visually distinct blobs don't align with any algorithm's actual cluster boundaries.

**Practical implication:** the 4-segment table above is a useful, actionable approximation — but an *imposed* partition of a spectrum, not a *discovered* set of natural customer types. Segment boundaries are soft; a customer near a boundary could reasonably sit in either neighboring group.

---

## 🗂️ Repo Structure

```
notebooks/
  01_eda_cleaning.ipynb      # Data loading, cleaning, RFM feature engineering
  02_modeling.ipynb          # PCA, K-Means, Hierarchical, DBSCAN, UMAP
```

> `data/` is gitignored and not part of this repo — see **How to Reproduce** below.

## ⚙️ How to Reproduce

1. Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) above and place it under `data/`
2. Run `notebooks/01_eda_cleaning.ipynb` — outputs `rfm.csv` and `feature_scaled.csv`
3. Run `notebooks/02_modeling.ipynb`

## 🗝️ Key Decisions Worth Noting

- Dropped ~22–25% of transactions with missing CustomerID (guest checkouts) — customer-level segmentation requires a known customer.
- Retained extreme high-Monetary customers (likely wholesale accounts) rather than removing them — used as a real test case for comparing how K-Means/Hierarchical (distance/centroid-based) handle outliers versus DBSCAN (density-based, designed to flag them as noise).

## ⚠️ Limitations

- Only Recency, Frequency, and Monetary were used — no product category, seasonality, or channel data.
- Champions' average Monetary is likely inflated by a handful of extreme wholesale-pattern accounts (see median vs. mean check in notebook).
- Returns/cancellations were excluded entirely rather than modeled as a behavioral signal.