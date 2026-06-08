from src.data_preprocessing import load_data
from src.rfm_analysis import create_rfm
from src.pca_analysis import apply_pca
from src.clustering import create_clusters

import pandas as pd
from src.visualization import (
    elbow_method,
    customer_clusters,
    pca_clusters
)

# Load Dataset
df = load_data("dataset/Mall_Customers.csv")

# Create RFM
rfm = create_rfm(df)

# PCA
pca_data = apply_pca(rfm)

# Clustering
labels, model = create_clusters(pca_data)

# Add labels
df["Cluster"] = labels

# Save Report
df.to_csv(
    "outputs/cluster_report.csv",
    index=False
)

elbow_method(rfm)

customer_clusters(
    pca_data,
    labels
)

pca_clusters(
    pca_data,
    labels
)
print("Customer Segmentation Completed")