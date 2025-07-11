import os
import subprocess
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm  # Progress bar
import matplotlib.pyplot as plt
from itertools import product
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 for 3D plotting

# Parameter discretizations
# Solar zenith angles
#minimum sza (at noon time) in Bergen is 35 according to https://en.wikipedia.org/wiki/Solar_zenith_angle 
SZA_VALUES = [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]  

# Surface albedo values
# 5, 25, 50, 75 and 95 percentiles, empirical values from S5P
ALBEDO_VALUES = [0.081, 0.129, 0.174, 0.224, 0.354] 

# Altitudes in km 
# Range in Bergen is 0 (sea level) - 800 with a peak around 50m. From Copernicus DEM 30m resolution
# 50%, 75% and 95% percentiles 
ALTITUDES = [0.08, 0.226, 0.521] 

# Effective droplet radius
# For ice clouds
IC_REFF = 20 # from libRadtran example 
# For water clouds 
WC_REFF = 10 # from libRadtran example

# Liquid/ice water content
LWC = 0.1 # from libRadtran simple water cloud example 
IWC = 0.015 # from libRadtran simple ice cloud example

# Cloud optical depth at 760nm, from Sentinel-5P
# Percentiles 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, empirical values from S5P
TAU_760 = [1.0, 3.41, 5.50, 7.68, 10.18]#, 13.67, 19.34, 27.79, 42.03, 73.23, 125.42, 250.0]
 
# Cloud base height in km, from Sentinel-5P
# Percentiles 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, empirical values from S5P 
CLOUD_BASE_HEIGHT = [0.052, 0.285]#, 0.571, 0.915, 1.286, 1.753, 2.370, 3.171, 4.165, 5.451, 6.543, 8.498]

# Cloud thickness (vertical extent) in km, from Sentinel-5P 
# fixed at 1km, according to satellite data valid for large majority of clouds 
CLOUD_VERTICAL_EXTENT = 1 

# Atmospheric profiles, provided by libRadtran
# april-september: use subarctic summer, else winter
ATM_PROFILES = ['subarctic_summer', 'subarctic_winter']

# COD scaling factor, conversion from 760nm to 550nm 
# tau_550 = tau_760 * (760/550)^alpha, alpha ~ 0.3 (Angstrom coefficient) = tau_760 * 1.1 
# TODO: backup this calculation with literature
TAU_SCALING_FACTOR = 1.1 

# Output directory 
OUTPUT_DIR = "output/LUT/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def descriptive_stats(df):
    """Generate descriptive statistics of cloud properties in Bergen"""
    print(df.describe())
    
    # Calculate percentiles
    # Define percentiles
    albedo_percentiles = [5, 25, 50, 75, 95]
    cloud_percentiles = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]

    # Compute and round percentiles
    albedo_stats = df["surface_albedo"].quantile([p / 100 for p in albedo_percentiles]).round(3)
    base_height_stats = df["cloud_base_height"].quantile([p / 100 for p in cloud_percentiles]).round(0)
    cod_stats = df["cloud_optical_depth"].quantile([p / 100 for p in cloud_percentiles]).round(2)

    # Print results
    print("Surface Albedo Percentiles:")
    print(albedo_stats)
    print("\nCloud Base Height Percentiles:")
    print(base_height_stats)
    print("\nCloud Optical Depth Percentiles:")
    print(cod_stats)
 
    # Plot 4 histograms in a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Distributions of Cloud and Surface Properties", fontsize=16)

    # Plot each feature
    axs[0, 0].hist(df["surface_albedo"], bins=50, color='skyblue', edgecolor='black')
    axs[0, 0].set_title("Surface Albedo")
    axs[0, 0].set_xlabel("Albedo")
    axs[0, 0].set_ylabel("Frequency")

    axs[0, 1].hist(df["cloud_base_height"], bins=50, color='lightgreen', edgecolor='black')
    axs[0, 1].set_title("Cloud Base Height (m)")
    axs[0, 1].set_xlabel("Height (m)")
    axs[0, 1].set_ylabel("Frequency")

    axs[1, 0].hist(df["cloud_vertical_extent"], bins=50, color='salmon', edgecolor='black')
    axs[1, 0].set_title("Cloud Vertical Extent (m)")
    axs[1, 0].set_xlabel("Extent (m)")
    axs[1, 0].set_ylabel("Frequency")

    axs[1, 1].hist(df["cloud_optical_depth"], bins=50, color='orange', edgecolor='black')
    axs[1, 1].set_title("Cloud Optical Depth")
    axs[1, 1].set_xlabel("Optical Depth")
    axs[1, 1].set_ylabel("Frequency")

    # Layout adjustment
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("output/s5p_cloud_properties_descriptive_stats.png")
    
def evaluate_cluster_range(df, k_range=range(2, 25), repeats=10, sample_size=10000):
    """Evaluate clustering quality for a range of cluster numbers."""
    
    # Prepare results
    silhouette_means = []
    silhouette_stds = []
    ch_means = []
    ch_stds = []
    db_means = []
    db_stds = []

    for k in k_range:
        print(f"----------------- k = {k} ------------------")
        sil_scores = []
        ch_scores = []
        db_scores = []

        for r in range(repeats):
            # New seed for each repeat
            seed = 10 + r  
            df_sample = df.sample(n=sample_size, random_state=seed)
            features = df_sample[["cloud_base_height", "cloud_vertical_extent", "cloud_optical_depth"]]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features)

            # KMeans clustering
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto")
            labels = kmeans.fit_predict(X_scaled)

            # Metrics
            sil = silhouette_score(X_scaled, labels)
            ch = calinski_harabasz_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)

            sil_scores.append(sil)
            ch_scores.append(ch)
            db_scores.append(db)
            
            print(f"Silhouette Score: {sil:.3f}")
            print(f"Calinski-Harabasz Score: {ch:.2f}")
            print(f"Davies-Bouldin Index: {db:.3f}\n")

        # Store mean and std
        silhouette_means.append(np.mean(sil_scores))
        silhouette_stds.append(np.std(sil_scores))
        ch_means.append(np.mean(ch_scores))
        ch_stds.append(np.std(ch_scores))
        db_means.append(np.mean(db_scores))
        db_stds.append(np.std(db_scores))

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    k_vals = list(k_range)

    axs[0].errorbar(k_vals, silhouette_means, yerr=silhouette_stds, fmt='-o', capsize=4, color='skyblue')
    axs[0].set_title("Silhouette Score vs. Cluster Count")
    axs[0].set_xlabel("Number of Clusters")
    axs[0].set_ylabel("Silhouette Score")

    axs[1].errorbar(k_vals, ch_means, yerr=ch_stds, fmt='-o', capsize=4, color='seagreen')
    axs[1].set_title("Calinski-Harabasz Score vs. Cluster Count")
    axs[1].set_xlabel("Number of Clusters")
    axs[1].set_ylabel("CH Score")

    axs[2].errorbar(k_vals, db_means, yerr=db_stds, fmt='-o', capsize=4, color='salmon')
    axs[2].set_title("Davies-Bouldin Index vs. Cluster Count")
    axs[2].set_xlabel("Number of Clusters")
    axs[2].set_ylabel("DB Index")

    plt.suptitle(f"Clustering Evaluation for k = {k_vals[0]} to {k_vals[-1]}  (avg of {repeats} runs per k)", fontsize=16)
    plt.tight_layout()
    plt.savefig("output/s5p_cloud_properties_evaluate_num_clusters.png")
    plt.show()
    
def generate_cloud_classes(df, n_clusters=4, sample_size=10000, random_state=14): 
    """Cluster clouds according to properties cloud_base_height, cloud_vertical_extent, cloud_optical_depth.""" 
    # Subsample
    df_sample = df.sample(n=sample_size, random_state=random_state)

    # Normalize features
    features = df_sample[["cloud_base_height", "cloud_vertical_extent", "cloud_optical_depth"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)
    
    # Evaluation metrics
    silhouette = silhouette_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)

    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Calinski-Harabasz Score: {ch_score:.2f}")
    print(f"Davies-Bouldin Index: {db_score:.3f}\n")

    # Centroids in original units
    centroids_original_units = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids_original_units, columns=features.columns)
    centroids_df.index.name = "Cluster"
    print("Cluster centroids (original feature space):")
    print(centroids_df.round(2))
    
    # Print size of each cluster
    cluster_sizes = np.bincount(labels)
    print("\nCluster sizes (number of points in each cluster):")
    for i, size in enumerate(cluster_sizes):
        print(f"Cluster {i}: {size} points")
        
    # Visualize using pca
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_plot = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_plot['Cluster'] = labels

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_plot['PC1'], df_plot['PC2'], c=df_plot['Cluster'], cmap='tab10', alpha=0.6, s=10)
    plt.title(f"PCA Projection of Cloud Clusters (k={n_clusters})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"output/s5p_cloud_properties_visualize_k={n_clusters}_pca.png")
    
    # Visualize 3D using all features 
    # Create a DataFrame with original (unscaled) values and cluster labels
    df_sample_plot = df_sample.copy()
    df_sample_plot["Cluster"] = labels

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 3D scatter plot
    scatter = ax.scatter(
        df_sample_plot["cloud_base_height"],
        df_sample_plot["cloud_vertical_extent"],
        df_sample_plot["cloud_optical_depth"],
        c=df_sample_plot["Cluster"],
        cmap='tab10',
        s=10,
        alpha=0.6
    )

    ax.set_xlabel("Cloud Base Height (m)")
    ax.set_ylabel("Vertical Extent (m)")
    ax.set_zlabel("Optical Depth")
    ax.set_title(f"3D Cluster Visualization of Clouds (k={n_clusters})")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc='upper right')
    ax.add_artist(legend)
    plt.tight_layout()
    plt.savefig(f"output/s5p_cloud_properties_visualize_clusters_k={n_clusters}_3D.png")
    plt.show()    

    return labels, centroids_df

def generate_LUT():
    os.makedirs("cloudfiles", exist_ok=True)
    
    outer_combos = list(product(SZA_VALUES, ALBEDO_VALUES, ALTITUDES, ATM_PROFILES))

    for sza, albedo, alt, profile in tqdm(outer_combos, desc="🌍 Generate LUT ", unit="config"):
         # Clear sky first
        clear_input = generate_uvspec_input(sza, albedo, alt, profile)
        ghi_clear, direct_clear, diffuse_clear = run_uvspec(clear_input)
        
        print(f"\n🔵 CLEAR SKY | Atmos={profile}, SZA={sza}, Albedo={albedo}, Alt={alt} km")
        print(f"    → GHI_clear={ghi_clear:.2f} W/m², Direct_clear={direct_clear:.2f}, Diffuse_clear={diffuse_clear:.2f}")

        results = []

        for base, tau760 in product(CLOUD_BASE_HEIGHT, TAU_760):            
            top = base + CLOUD_VERTICAL_EXTENT
            tau550 = np.round(TAU_SCALING_FACTOR * tau760,3)

            # Determine cloud phase
            is_ice = base >= 3 and tau550 <= 5
            cloud_type = "ice" if is_ice else "water"
            reff = IC_REFF if is_ice else WC_REFF
            lwc = IWC if is_ice else LWC

            cloud_file = f"cloudfiles/{cloud_type}_base{base*1000}_tau{round(tau550)}.dat"
            write_cloud_file(cloud_file, base, top, lwc, reff)

            # Cloudy input with cloud file and modify tau550
            cloudy_input = generate_uvspec_input(sza, albedo, alt, profile,
                                                 cloud_file=cloud_file,
                                                 cloud_type=cloud_type,
                                                 tau550=tau550)
            ghi_cloudy, direct_cloudy, diffuse_cloudy = run_uvspec(cloudy_input)

            # CAF = cloudy / clear
            caf_ghi = ghi_cloudy / ghi_clear if ghi_clear > 0 else np.nan
            caf_dir = direct_cloudy / direct_clear if direct_clear > 0 else np.nan
            caf_dif = diffuse_cloudy / diffuse_clear if diffuse_clear > 0 else np.nan
            
            print(f"☁️  CLOUDY SKY | Base={base} km, Type={cloud_type}, Tau550={tau550}")
            print(f"    → GHI_cloudy={ghi_cloudy:.2f} W/m², Direct_cloudy={direct_cloudy:.2f}, Diffuse_cloudy={diffuse_cloudy:.2f}")
            print(f"📉 CAF        | CAF_GHI={caf_ghi:.3f}, CAF_Direct={caf_dir:.3f}, CAF_Diffuse={caf_dif:.3f}")


            results.append([
                profile, sza, albedo, alt,
                base, top, tau550, cloud_type,
                ghi_clear, direct_clear, diffuse_clear,
                ghi_cloudy, direct_cloudy, diffuse_cloudy,
                caf_ghi, caf_dir, caf_dif
            ])
            
        # File naming convention
        filename = f"LUT_{profile}_sza{sza}_alb{albedo}_alt{alt}.csv"
        output_path = os.path.join(OUTPUT_DIR, filename.replace(" ", "_"))

        # Write this batch to CSV
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Atmosphere", "SZA", "Albedo", "Altitude_km",
                "CloudBase_km", "CloudTop_km", "Tau550", "CloudType",
                "GHI_clear", "Direct_clear", "Diffuse_clear",
                "GHI_cloudy", "Direct_cloudy", "Diffuse_cloudy",
                "CAF_GHI", "CAF_Direct", "CAF_Diffuse"
            ])
            writer.writerows(results)

        print(f"✅ Saved LUT: {output_path}")

   

def write_cloud_file(filename, base, top, lwc, reff):
    with open(filename, "w") as f:
        f.write("# z LWC/IWC R_eff\n")
        f.write(f"{top:.3f} 0 0\n")
        f.write(f"{base:.3f} {lwc} {reff}\n")

def generate_uvspec_input(sza, albedo, altitude, atmosphere, cloud_file=None, cloud_type=None, tau550=None):
    lines = [
        "verbose",
        f"atmosphere_file {atmosphere}",
        "rte_solver disort",
        "number_of_streams 16",
        "wavelength 250 2500",
        "mol_abs_param KATO",
        "output_user eglo edir edn",
        "output_process sum",
        f"sza {sza}",
        f"albedo {albedo}",
        f"altitude {altitude}"
    ]

    if cloud_file and cloud_type and tau550:
        if cloud_type == "ice":
            lines.append(f"ic_file 1D {cloud_file}")
            lines.append(f"ic_modify tau550 set {tau550}")
        else:
            lines.append(f"wc_file 1D {cloud_file}")
            lines.append(f"wc_modify tau550 set {tau550}")

    return "\n".join(lines)

def cloud_file_alt_range(filepath):
    try:
        with open(filepath) as f:
            lines = [line for line in f if not line.startswith("#") and line.strip()]
            altitudes = [float(line.split()[0]) for line in lines]
            z_max, z_min = max(altitudes), min(altitudes)
            return f"{z_min} {z_max}"
    except Exception as e:
        print(f"Failed to read cloud file {filepath}: {e}")
        return "0 0"

def run_uvspec(input_str, out_path="output/uvspec.out", err_path="output/verbose.txt"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as out_file, open(err_path, "w") as err_file:
        result = subprocess.run(
            ["uvspec"],
            input=input_str,
            text=True,
            stdout=out_file,
            stderr=err_file
        )

    if result.returncode != 0:
        print(f"uvspec error. See {err_path}")
        return np.nan, np.nan, np.nan

    try:
        with open(out_path, "r") as f:
            for line in f:
                if not line.startswith("#") and line.strip():
                    wl, edir, edn = map(float, line.split()[:3])
                    eglo = edir + edn
                    return eglo, edir, edn
        return np.nan, np.nan, np.nan
    except Exception as e:
        print(f"Failed to parse output: {e}")
        return np.nan, np.nan, np.nan

if __name__ == "__main__":
    #df = pd.read_csv("data/s5p_cloud_properties_per_pixel.csv")
    #df = df[["surface_albedo", "cloud_base_height", "cloud_top_height", "cloud_optical_depth"]].dropna()
    #df["cloud_vertical_extent"] = df["cloud_top_height"] - df["cloud_base_height"]
    #df = df.drop(columns=["cloud_top_height"])
    
    #descriptive_stats(df)
    #evaluate_cluster_range(df)
    #labels, centroids = generate_cloud_classes(df, n_clusters=18)
    generate_LUT()