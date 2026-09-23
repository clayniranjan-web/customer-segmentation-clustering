# Customer Segmentation using Unsupervised Learning

**Status:** 🚧 In Progress — Notebook 1 (data cleaning + feature engineering) complete. Notebook 2 (clustering: PCA, K-Means, Hierarchical, DBSCAN, UMAP) in progress.

## Problem Statement

Segment an online retailer's ~1M transactions into distinct customer groups based on purchasing behavior (Recency, Frequency, Monetary), using unsupervised clustering — since no labeled "customer type" exists — so the business can identify high-value customers, at-risk/churning customers, and outliers, instead of treating every customer the same.

## Dataset

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — UCI Machine Learning Repository. ~1,067,371 transactions across two sheets (Dec 2009 – Dec 2011), UK-based online retailer. Not included in this repo (see below) — download directly from the source link above.

## Approach

1. **Data cleaning** — removed transactions with missing CustomerID, exact duplicate rows, zero-price entries, and cancelled orders (negative quantity).
2. **Feature engineering** — built RFM (Recency, Frequency, Monetary) features per customer via aggregation from transaction-level data.
3. **Preprocessing** — log-transformed Monetary and Frequency to address right-skew from outlier (likely wholesale) customers; left Recency untransformed after confirming its distribution is multi-modal rather than skewed. Scaled all features with StandardScaler.
4. **Clustering (in progress)** — comparing K-Means, Hierarchical, and DBSCAN on the same feature set, with PCA and UMAP used for dimensionality reduction and visualization.

## PCA
The PCA projection shows a continuous, dense cloud without clearly separated clusters, suggesting customer behavior exists on a spectrum rather than in distinct categories. A small number of extreme outliers (high Monetary/Frequency) are visible, consistent with the wholesale-pattern customers identified during EDA

## Repo Structure

```
notebooks/
  01_eda_cleaning.ipynb      # Data loading, cleaning, RFM feature engineering
  02_modeling.ipynb          # PCA, K-Means, Hierarchical, DBSCAN, UMAP
data/                        # Not tracked — see Dataset section above
```

## How to Reproduce

1. Download the dataset from the UCI link above and place it under `data/`
2. Run `notebooks/01_eda_cleaning.ipynb` — outputs `rfm.csv` and `feature_scaled.csv`
3. Run `notebooks/02_modeling.ipynb`

## Key Decisions Worth Noting

- Dropped ~22–25% of transactions due to missing CustomerID (guest checkouts) — customer-level segmentation requires a known customer.
- Retained extreme high-Monetary customers (likely wholesale accounts) rather than removing them — used as a real test case for comparing how K-Means/Hierarchical (distance/centroid-based) handle outliers versus DBSCAN (density-based, designed to flag them as noise).