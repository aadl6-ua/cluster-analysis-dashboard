"""
Clustering algorithms module
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage


def kmeans_clustering(data, n_clusters=3, random_state=42):
    """
    Perform K-means clustering
    
    Returns:
    - model: Fitted KMeans model
    - labels: Cluster labels
    - metrics: Dictionary with evaluation metrics
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(data)
    
    metrics = calculate_metrics(data, labels)
    metrics['inertia'] = model.inertia_
    metrics['n_clusters'] = n_clusters
    
    return model, labels, metrics


def find_optimal_k(data, max_k=10, min_k=2):
    """Find optimal number of clusters using elbow method and silhouette score"""
    inertias = []
    silhouettes = []
    k_range = range(min_k, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        inertias.append(kmeans.inertia_)
        
        if k > 1:
            silhouettes.append(silhouette_score(data, labels))
        else:
            silhouettes.append(0)
    
    # Find elbow point (simplified)
    best_k_silhouette = k_range[np.argmax(silhouettes)]
    
    return {
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouettes': silhouettes,
        'best_k': best_k_silhouette
    }


def dbscan_clustering(data, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering
    
    Returns:
    - model: Fitted DBSCAN model
    - labels: Cluster labels
    - metrics: Dictionary with evaluation metrics
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    metrics = {}
    if n_clusters > 1:
        # Only calculate metrics if we have valid clusters
        valid_mask = labels != -1
        if valid_mask.sum() > 0:
            metrics = calculate_metrics(data[valid_mask], labels[valid_mask])
    
    metrics['n_clusters'] = n_clusters
    metrics['n_noise'] = n_noise
    metrics['noise_ratio'] = n_noise / len(labels) if len(labels) > 0 else 0
    
    return model, labels, metrics


def hierarchical_clustering(data, n_clusters=3, linkage_method='ward'):
    """
    Perform Hierarchical clustering
    
    Returns:
    - model: Fitted AgglomerativeClustering model
    - labels: Cluster labels
    - metrics: Dictionary with evaluation metrics
    - linkage_matrix: Linkage matrix for dendrogram
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(data)
    
    # Calculate linkage matrix for dendrogram
    linkage_matrix = linkage(data, method=linkage_method)
    
    metrics = calculate_metrics(data, labels)
    metrics['n_clusters'] = n_clusters
    
    return model, labels, metrics, linkage_matrix


def calculate_metrics(data, labels):
    """Calculate clustering evaluation metrics"""
    metrics = {}
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters > 1 and len(data) > n_clusters:
        try:
            metrics['silhouette'] = silhouette_score(data, labels)
        except:
            metrics['silhouette'] = None
        
        try:
            metrics['davies_bouldin'] = davies_bouldin_score(data, labels)
        except:
            metrics['davies_bouldin'] = None
        
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(data, labels)
        except:
            metrics['calinski_harabasz'] = None
    else:
        metrics['silhouette'] = None
        metrics['davies_bouldin'] = None
        metrics['calinski_harabasz'] = None
    
    return metrics


def compare_algorithms(data, kmeans_labels, dbscan_labels, hierarchical_labels):
    """Compare metrics across different algorithms"""
    comparison = {
        'K-means': calculate_metrics(data, kmeans_labels),
        'DBSCAN': calculate_metrics(data[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1]) if (dbscan_labels != -1).any() else {},
        'Hierarchical': calculate_metrics(data, hierarchical_labels)
    }
    
    # Add cluster counts
    comparison['K-means']['n_clusters'] = len(set(kmeans_labels))
    comparison['DBSCAN']['n_clusters'] = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    comparison['Hierarchical']['n_clusters'] = len(set(hierarchical_labels))
    
    return comparison
