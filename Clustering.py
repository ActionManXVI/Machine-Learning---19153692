# combined_store_location_script.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from DataPreprocessing import DataPreprocessor
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# Set the working directory
os.chdir(os.path.dirname(__file__))

def perform_clustering(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters
    silhouette_scores = []
    K = range(2, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    optimal_k = K[np.argmax(silhouette_scores)]
    
    # Perform K-means clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    return clusters, optimal_k

def visualize_clusters_on_map(df, clusters):
    df['clusters'] = clusters
    df['storeClass'] = df['storeClass'].astype(str).replace({'F': 'Flagship', 'nan': 'N/A'})

    # Create a custom colormap
    colors = ['#FF0000', '#ff7f0e', '#0000FF', '#9467bd']  # Red, Flagship, Blue, F, N/A
    n_bins = len(df['storeClass'].unique())
    cmap = ListedColormap(colors[:n_bins])

    # Set up the plot with a specific map projection
    fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add country boundaries
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)

    # Create a scatter plot of store locations
    scatter = ax.scatter(df['longitude'], df['latitude'], c=clusters, 
                         cmap=cmap, s=20, alpha=0.7, transform=ccrs.PlateCarree())

    # Customize the plot
    ax.set_global()
    ax.set_title('Store Locations Clustered Worldwide', fontsize=16)

    # Add a color bar legend
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Cluster')

    # Create a custom legend for store classes
    unique_classes = df['storeClass'].unique()
    legend_elements = [plt.scatter([], [], c=[cmap(i/len(unique_classes))], label=cls) 
                       for i, cls in enumerate(unique_classes)]
    ax.legend(handles=legend_elements, title="Store Classes", loc="lower left")

    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    preprocessor = DataPreprocessor('hm_all_stores.csv')
    df = preprocessor.preprocess()
    X = preprocessor.get_cluster_data()
    
    clusters, optimal_k = perform_clustering(X)
    print(f"Optimal number of clusters: {optimal_k}")
    
    visualize_clusters_on_map(df, clusters)
