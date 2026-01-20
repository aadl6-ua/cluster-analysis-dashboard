"""
Visualization utilities using Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram


def plot_clusters_2d(data, labels, title="Cluster Visualization", algorithm=""):
    """Create 2D scatter plot of clusters using first 2 principal components"""
    
    # Use first two columns for visualization
    df_plot = pd.DataFrame({
        'Feature 1': data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data[:, 0],
        'Feature 2': data.iloc[:, 1] if isinstance(data, pd.DataFrame) else data[:, 1],
        'Cluster': labels.astype(str)
    })
    
    # Color mapping
    unique_clusters = sorted(df_plot['Cluster'].unique())
    color_map = {str(c): px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)] 
                 for i, c in enumerate(unique_clusters)}
    
    # Handle noise points (DBSCAN)
    if '-1' in color_map:
        color_map['-1'] = '#999999'
    
    fig = px.scatter(
        df_plot, 
        x='Feature 1', 
        y='Feature 2', 
        color='Cluster',
        title=f"{title} ({algorithm})",
        color_discrete_map=color_map,
        labels={'Cluster': 'Cluster ID'}
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(
        height=500,
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig


def plot_clusters_3d(data, labels, title="3D Cluster Visualization", algorithm=""):
    """Create 3D scatter plot of clusters"""
    
    # Use first three columns
    n_features = data.shape[1]
    
    df_plot = pd.DataFrame({
        'Feature 1': data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data[:, 0],
        'Feature 2': data.iloc[:, 1] if isinstance(data, pd.DataFrame) else data[:, 1],
        'Feature 3': data.iloc[:, min(2, n_features-1)] if isinstance(data, pd.DataFrame) else data[:, min(2, n_features-1)],
        'Cluster': labels.astype(str)
    })
    
    fig = px.scatter_3d(
        df_plot,
        x='Feature 1',
        y='Feature 2',
        z='Feature 3',
        color='Cluster',
        title=f"{title} ({algorithm})",
        labels={'Cluster': 'Cluster ID'}
    )
    
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(height=600, template='plotly_white')
    
    return fig


def plot_elbow_silhouette(optimization_results):
    """Plot elbow and silhouette curves"""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Elbow Method', 'Silhouette Score'),
        horizontal_spacing=0.12
    )
    
    k_range = optimization_results['k_range']
    inertias = optimization_results['inertias']
    silhouettes = optimization_results['silhouettes']
    best_k = optimization_results['best_k']
    
    # Elbow plot
    fig.add_trace(
        go.Scatter(
            x=k_range, 
            y=inertias, 
            mode='lines+markers',
            name='Inertia',
            line=dict(color='#636EFA', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Silhouette plot
    fig.add_trace(
        go.Scatter(
            x=k_range, 
            y=silhouettes, 
            mode='lines+markers',
            name='Silhouette',
            line=dict(color='#EF553B', width=3),
            marker=dict(size=8)
        ),
        row=1, col=2
    )
    
    # Mark optimal k
    if best_k in k_range:
        idx = k_range.index(best_k)
        fig.add_trace(
            go.Scatter(
                x=[best_k],
                y=[silhouettes[idx]],
                mode='markers',
                name=f'Optimal k={best_k}',
                marker=dict(size=15, color='green', symbol='star'),
                showlegend=True
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=2)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    
    fig.update_layout(
        height=400,
        showlegend=True,
        template='plotly_white',
        title_text="K-Means Optimization"
    )
    
    return fig


def plot_dendrogram(linkage_matrix, n_samples=None):
    """Plot hierarchical clustering dendrogram"""
    
    # Create dendrogram
    dend = dendrogram(linkage_matrix, no_plot=True, truncate_mode='lastp', p=30)
    
    # Extract data for plotly
    icoord = np.array(dend['icoord'])
    dcoord = np.array(dend['dcoord'])
    
    fig = go.Figure()
    
    # Add dendrogram lines
    for i in range(len(icoord)):
        fig.add_trace(go.Scatter(
            x=icoord[i],
            y=dcoord[i],
            mode='lines',
            line=dict(color='#636EFA', width=2),
            hoverinfo='skip',
            showlegend=False
        ))
    
    fig.update_layout(
        title='Hierarchical Clustering Dendrogram',
        xaxis=dict(title='Sample Index', showticklabels=False),
        yaxis=dict(title='Distance'),
        height=500,
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig


def plot_metrics_comparison(comparison_dict):
    """Plot comparison of metrics across algorithms"""
    
    algorithms = list(comparison_dict.keys())
    metrics_names = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Silhouette Score<br>(higher is better)', 
                       'Davies-Bouldin Index<br>(lower is better)',
                       'Calinski-Harabasz Score<br>(higher is better)'),
        horizontal_spacing=0.1
    )
    
    colors = {'K-means': '#636EFA', 'DBSCAN': '#EF553B', 'Hierarchical': '#00CC96'}
    
    for col, metric in enumerate(metrics_names, 1):
        values = []
        alg_names = []
        bar_colors = []
        
        for alg in algorithms:
            if metric in comparison_dict[alg] and comparison_dict[alg][metric] is not None:
                values.append(comparison_dict[alg][metric])
                alg_names.append(alg)
                bar_colors.append(colors.get(alg, '#AB63FA'))
        
        if values:
            fig.add_trace(
                go.Bar(
                    x=alg_names,
                    y=values,
                    marker_color=bar_colors,
                    showlegend=False,
                    text=[f'{v:.3f}' for v in values],
                    textposition='outside'
                ),
                row=1, col=col
            )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        title_text="Algorithm Performance Comparison"
    )
    
    return fig


def plot_cluster_sizes(labels, algorithm=""):
    """Plot distribution of cluster sizes"""
    
    unique, counts = np.unique(labels, return_counts=True)
    
    df = pd.DataFrame({
        'Cluster': [f'Cluster {i}' if i != -1 else 'Noise' for i in unique],
        'Size': counts
    })
    
    fig = px.bar(
        df,
        x='Cluster',
        y='Size',
        title=f'Cluster Size Distribution ({algorithm})',
        color='Size',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig
