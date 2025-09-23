#Cluster the pixels in Bergen ROI according to seasonal cloudiness profiles

import rasterio
import cartopy.crs as ccrs
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.utils import resample
import matplotlib.pyplot as plt
from time import time
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import pandas as pd
from rasterio.plot import show


# mapping from feature to band name
feature_names = [
    'mean_winter', 'mean_spring', 'mean_summer', 'mean_autumn',
    'mean_jan', 'mean_feb', 'mean_mar', 'mean_apr', 'mean_may', 'mean_jun',
    'mean_jul', 'mean_aug', 'mean_sep', 'mean_oct', 'mean_nov', 'mean_dec',
    'q1_alltime', 'q1_jan', 'q1_feb', 'q1_mar', 'q1_apr', 'q1_may', 'q1_jun',
    'q1_jul', 'q1_aug', 'q1_sep', 'q1_oct', 'q1_nov', 'q1_dec',
    'q1_winter', 'q1_spring', 'q1_summer', 'q1_autumn',
    'median_alltime', 'median_jan', 'median_feb', 'median_mar', 'median_apr',
    'median_may', 'median_jun', 'median_jul', 'median_aug', 'median_sep',
    'median_oct', 'median_nov', 'median_dec', 'median_winter', 'median_spring',
    'median_summer', 'median_autumn', 'mean_alltime'
]

cluster_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',  '#991122']  # red, green, yellow, blue, orange


def compute_pseudo_inertia(X, labels, centroids):
    "Compute pseudo inertia value for elbow method"
    inertia_value = 0.0
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        if cluster_points.shape[0] > 0:
            # Compute squared distances manually (no large intermediate arrays)
            diff = cluster_points - centroids[i]
            sq_dist = np.einsum('ij,ij->i', diff, diff)  # faster and memory-efficient
            inertia_value += np.sum(sq_dist)
    return inertia_value

def plot_elbow_silhouette(X, mask): 
    """compute and plot elbow and silhouette score"""
    # Elbow metric (inertia)
    inertia = []
    silhouette = []
    ks = [2, 3, 4, 5, 6]

    seed = np.random.randint(0, 1000)
    print(f"Seed for sample generation: {seed}")

    for k in ks:
        with rasterio.open(f"data/kmeans_clusters_k{k}_trainsize100000.tif") as src:
            labels = src.read(1).flatten()[mask]  # nan mask
            
        # Compute pseudo inertia
        km = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=1)
        km.cluster_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        X_sample_big, labels_sample_big = resample(X, labels, n_samples=1_000_000, random_state=seed)
        X_sample_small, labels_sample_small = resample(X, labels, n_samples=10_000, random_state=seed)

        print(f"k {k}")
        start_inertia = time()
        inertia_value = compute_pseudo_inertia(X_sample_big, labels_sample_big, km.cluster_centers)
        print(f" inertia value {inertia_value}, computation time: {np.round(time() - start_inertia, 3)} seconds")
        inertia.append(inertia_value)
        
        # Silhouette score
        start_silhouette = time()
        silhouette.append(silhouette_score(X_sample_small, labels_sample_small))
        print(f" silhouette {silhouette[-1]}, computation time: {np.round(time()- start_silhouette, 3)} seconds")

    # Plot Elbow
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ks, inertia, 'o-')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")

    # Plot Silhouette
    plt.subplot(1, 2, 2)
    plt.plot(ks, silhouette, 'o-')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis")

    plt.tight_layout()
    plt.savefig("output/elbow_method_silhouette_score.png")
    
# Load and plot function
def plot_cluster_tif(path, k, title):
    with rasterio.open(path) as src:
        print(f"transform (k={k}) : {src.transform}")
        data = src.read(1)
        fig, ax = plt.subplots(figsize=(8, 8))
        cmap = ListedColormap(cluster_colors[:k])
        rasterio.plot.show(data, ax=ax, transform=src.transform, cmap=cmap)
        
        ax.set_title(title)
        ax.axis('off')

        # Legend
        unique_clusters = np.unique(data)
        unique_clusters = unique_clusters[unique_clusters >= 0]
        legend_elements = [
            Patch(facecolor=cluster_colors[c], label=f"Cluster {c}") for c in unique_clusters
        ]
        ax.legend(handles=legend_elements, loc='upper right', title="Clusters")

        plt.savefig(f"output/kmeans_clusters_k{k}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
start = time()

plot_cluster_tif("data/kmeans_clusters_k3_trainsize100000.tif", 3, "KMeans Clusters (k=3)")
plot_cluster_tif("data/kmeans_clusters_k4_trainsize100000.tif", 4, "KMeans Clusters (k=4)")

""" # Load 51-band image (e.g., already stacked as a single multi-band GeoTIFF)
with rasterio.open("data/features_multiband_image_rescaled.tif") as src:
    features = src.read()  # shape: (51, height, width)
    print(f"Features shape: {features.shape}")

print(f"loading combined image takes {np.round(time() - start, 3)} seconds")

# Reshape
X = features.reshape(51, -1).T  # shape: (n_pixels, 51)
print(f"X shape: {X.shape}")

# Remove masked/invalid pixels (e.g., 0 or NaNs)
mask = ~np.any(np.isnan(X), axis=1)
X = X[mask]

#plot_elbow_silhouette(X, mask)
# Convert X to a DataFrame
df = pd.DataFrame(X, columns=[f"{feature_names[i]}" for i in range(X.shape[1])])

# Add cluster labels for k=3 and k=4
with rasterio.open("data/kmeans_clusters_k3_trainsize100000.tif") as src_k3:
    data_k3 = src_k3.read(1)
    print(f"bounds: {src_k3.bounds}")
    print("Unique values in data_k3:", np.unique(data_k3))
    print("Shape of data_k3:", data_k3.shape)
    labels_k3 = data_k3.flatten()[mask]
    df["cluster_k3"] = labels_k3

with rasterio.open("data/kmeans_clusters_k4_trainsize100000.tif") as src_k4:
    data_k4 = src_k4.read(1)
    plot_cluster_tif(data_k4, k=4, title="KMeans Clusters (k=4)", bounds=src_k4.bounds)
    labels_k4 = data_k4.flatten()[mask]
    df["cluster_k4"] = labels_k4
 """
""" # Compute per-cluster means
cluster_means_k3 = df.groupby('cluster_k3').mean()
cluster_means_k4 = df.groupby('cluster_k4').mean()

# Compute standard deviation or range across cluster means
feature_std_across_k3 = cluster_means_k3.std()
feature_std_across_k4 = cluster_means_k4.std()
top_features_k3_std = feature_std_across_k3.sort_values(ascending=False).head(5)
top_features_k4_std = feature_std_across_k4.sort_values(ascending=False).head(5)

print(f"Top features k3: \n{top_features_k3_std}")
print(f"Top features k4: \n{top_features_k4_std}")

# now as a list
top_features_k3 = ['q1_winter', 'q1_spring', 'q1_summer', 'q1_autumn', 'q1_alltime'] #top_features_k3_std.index.tolist()
top_features_k4 = ['q1_winter', 'q1_spring', 'q1_summer', 'q1_autumn', 'q1_alltime'] #top_features_k4_std.index.tolist()

sample_df = df.sample(n=100000, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

for i, (cluster_col, top_features, ax, title) in enumerate([
    ("cluster_k3", top_features_k3, axes[0], "Top 5 Features by Cluster (k=3)"),
    ("cluster_k4", top_features_k4, axes[1], "Top 5 Features by Cluster (k=4)")
]):
    cluster_ids = sorted(sample_df[cluster_col].unique())
    n_clusters = len(cluster_ids)

    # Track positions and labels for x-axis
    positions = []
    xtick_labels = []
    legend_handles = []

    for j, feature in enumerate(top_features):
        for k, cluster in enumerate(cluster_ids):
            cluster_data = sample_df[sample_df[cluster_col] == cluster][feature]
            pos = j * (n_clusters + 1) + k
            positions.append(pos)

            # Boxplot with specific color
            bp = ax.boxplot(cluster_data, positions=[pos], widths=0.6,
                            patch_artist=True, boxprops=dict(facecolor=cluster_colors[cluster], color='black'),
                            medianprops=dict(color='black'))

            # Add only one label per feature
            if k == n_clusters // 2:
                xtick_labels.append(feature)
            elif k == n_clusters - 1:
                # append two because there is an empty tick between features
                xtick_labels.append("")
                xtick_labels.append("")
            else: 
                xtick_labels.append("")

        # Add one legend entry per cluster
        if j == 0:
            for k, cluster in enumerate(cluster_ids):
                legend_handles.append(plt.Line2D([], [], color=cluster_colors[cluster], marker='s',
                                                 linestyle='', markersize=10, label=f"Cluster {cluster}"))

    ax.set_xticks(np.arange(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
    ax.set_title(title)
    ax.set_ylabel("Feature Value")
    ax.legend(handles=legend_handles, title="Clusters", loc='upper right')

plt.tight_layout()
plt.savefig("output/boxplot_season_alltime_features_k3_k4.png", dpi=300)
 """

print(f"total runtime: {np.round(time() - start, 3)} seconds")