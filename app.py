"""
Interactive Clustering Analysis Dashboard
Developed by Álvaro Andrés De Lamo
University of Silesia - Cluster Analysis Course
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.preprocessing import load_data, preprocess_data, get_feature_summary
from utils.clustering import (
    kmeans_clustering, dbscan_clustering, hierarchical_clustering,
    find_optimal_k, compare_algorithms
)
from utils.visualization import (
    plot_clusters_2d, plot_clusters_3d, plot_elbow_silhouette,
    plot_dendrogram, plot_metrics_comparison, plot_cluster_sizes
)

# Page configuration
st.set_page_config(
    page_title="Cluster Analysis Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    .st-emotion-cache-16idsys p {
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None


def main():
    # Header
    st.title("🔬 Interactive Clustering Analysis Dashboard")
    st.markdown("**Analyze your data with K-means, DBSCAN, and Hierarchical Clustering**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/_static/favicon.png", width=80)
        st.title("⚙️ Configuration")
        
        # Data upload
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="Upload a CSV file with your dataset"
        )
        
        # Example data option
        use_example = st.checkbox("Use example dataset (Absenteeism at Work)", value=True)
        
        # Load data
        if uploaded_file is not None:
            data, error = load_data(uploaded_file)
            if error:
                st.error(f"Error loading file: {error}")
                return
            st.session_state.data_loaded = True
            st.success("✅ Data loaded successfully!")
        elif use_example:
            try:
                data = pd.read_csv('data/Absenteeism_at_work.csv', sep=';')
                if 'ID' in data.columns:
                    data = data.drop('ID', axis=1)
                st.session_state.data_loaded = True
                st.info("📊 Using example dataset")
            except Exception as e:
                st.error(f"Error loading example data: {e}")
                return
        else:
            st.warning("⚠️ Please upload a CSV file or use the example dataset")
            return
        
        st.markdown("---")
        
        # Preprocessing options
        st.subheader("🔧 Preprocessing")
        remove_outliers = st.checkbox("Remove outliers (IQR method)", value=True)
        outlier_threshold = st.slider(
            "Outlier threshold (IQR multiplier)",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            disabled=not remove_outliers
        )
        
        st.markdown("---")
        
        # Algorithm selection
        st.subheader("🎯 Algorithm Selection")
        algorithm = st.selectbox(
            "Choose clustering algorithm",
            ["K-means", "DBSCAN", "Hierarchical", "Compare All"]
        )
        
        st.markdown("---")
        
        # Algorithm parameters
        st.subheader("📊 Parameters")
        
        if algorithm in ["K-means", "Compare All"]:
            st.markdown("**K-means Settings**")
            auto_k = st.checkbox("Auto-detect optimal k", value=False)
            if auto_k:
                max_k = st.slider("Maximum k to test", 2, 15, 10)
                n_clusters = None
            else:
                n_clusters = st.slider("Number of clusters (k)", 2, 10, 3)
        
        if algorithm in ["DBSCAN", "Compare All"]:
            st.markdown("**DBSCAN Settings**")
            eps = st.slider("Epsilon (ε)", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("Min samples", 2, 20, 5)
        
        if algorithm in ["Hierarchical", "Compare All"]:
            st.markdown("**Hierarchical Settings**")
            n_clusters_hier = st.slider("Number of clusters", 2, 10, 3, key="hier_k")
            linkage_method = st.selectbox(
                "Linkage method",
                ["ward", "complete", "average", "single"]
            )
        
        st.markdown("---")
        
        # Run button
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Main content
    if not st.session_state.data_loaded:
        st.info("👈 Upload a CSV file or use the example dataset from the sidebar to begin")
        return
    
    # Display data info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", data.shape[0])
    with col2:
        st.metric("Columns", data.shape[1])
    with col3:
        st.metric("Missing Values", data.isnull().sum().sum())
    with col4:
        st.metric("Numeric Features", len(data.select_dtypes(include=[np.number]).columns))
    
    # Data preview
    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(data.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Statistical Summary")
            st.dataframe(data.describe(), use_container_width=True)
        with col2:
            st.subheader("Data Types")
            dtypes_df = pd.DataFrame({
                'Column': data.columns,
                'Type': data.dtypes.values,
                'Non-Null Count': data.count().values
            })
            st.dataframe(dtypes_df, use_container_width=True)
    
    if run_analysis:
        with st.spinner('Processing data and running clustering...'):
            # Preprocess data
            data_normalized, data_processed, preprocessing_info = preprocess_data(
                data,
                remove_outliers=remove_outliers,
                outlier_threshold=outlier_threshold
            )
            
            st.session_state.processed_data = {
                'normalized': data_normalized,
                'processed': data_processed,
                'info': preprocessing_info
            }
            
            # Show preprocessing results
            st.subheader("🔧 Preprocessing Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Samples", preprocessing_info['original_shape'][0])
            with col2:
                st.metric("Outliers Removed", preprocessing_info['outliers_removed'])
            with col3:
                st.metric("Final Samples", preprocessing_info['final_shape'][0])
            
            st.markdown("---")
            
            # Run clustering based on selected algorithm
            if algorithm == "K-means":
                st.subheader("📊 K-means Clustering Results")
                
                # Auto-detect optimal k if enabled
                if auto_k:
                    with st.spinner('Finding optimal k...'):
                        opt_results = find_optimal_k(data_normalized, max_k=max_k)
                        n_clusters = opt_results['best_k']
                        
                        st.info(f"✨ Optimal k detected: {n_clusters}")
                        
                        # Show optimization plots
                        fig_opt = plot_elbow_silhouette(opt_results)
                        st.plotly_chart(fig_opt, use_container_width=True)
                
                # Run K-means
                model, labels, metrics = kmeans_clustering(data_normalized, n_clusters=n_clusters)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Clusters", metrics['n_clusters'])
                with col2:
                    st.metric("Silhouette Score", f"{metrics['silhouette']:.3f}" if metrics['silhouette'] else "N/A")
                with col3:
                    st.metric("Davies-Bouldin", f"{metrics['davies_bouldin']:.3f}" if metrics['davies_bouldin'] else "N/A")
                with col4:
                    st.metric("Inertia", f"{metrics['inertia']:.2f}")
                
                # Visualizations
                tab1, tab2, tab3 = st.tabs(["2D Clusters", "3D Clusters", "Cluster Sizes"])
                
                with tab1:
                    fig_2d = plot_clusters_2d(data_normalized, labels, algorithm="K-means")
                    st.plotly_chart(fig_2d, use_container_width=True)
                
                with tab2:
                    fig_3d = plot_clusters_3d(data_normalized, labels, algorithm="K-means")
                    st.plotly_chart(fig_3d, use_container_width=True)
                
                with tab3:
                    fig_sizes = plot_cluster_sizes(labels, algorithm="K-means")
                    st.plotly_chart(fig_sizes, use_container_width=True)
                
                # Export results
                st.markdown("---")
                results_df = data_processed.copy()
                results_df['Cluster'] = labels
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="kmeans_results.csv",
                    mime="text/csv"
                )
            
            elif algorithm == "DBSCAN":
                st.subheader("📊 DBSCAN Clustering Results")
                
                # Run DBSCAN
                model, labels, metrics = dbscan_clustering(data_normalized, eps=eps, min_samples=min_samples)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Clusters Found", metrics['n_clusters'])
                with col2:
                    st.metric("Noise Points", metrics['n_noise'])
                with col3:
                    st.metric("Noise Ratio", f"{metrics['noise_ratio']*100:.1f}%")
                with col4:
                    if metrics.get('silhouette'):
                        st.metric("Silhouette Score", f"{metrics['silhouette']:.3f}")
                    else:
                        st.metric("Silhouette Score", "N/A")
                
                # Visualizations
                tab1, tab2, tab3 = st.tabs(["2D Clusters", "3D Clusters", "Cluster Sizes"])
                
                with tab1:
                    fig_2d = plot_clusters_2d(data_normalized, labels, algorithm="DBSCAN")
                    st.plotly_chart(fig_2d, use_container_width=True)
                
                with tab2:
                    fig_3d = plot_clusters_3d(data_normalized, labels, algorithm="DBSCAN")
                    st.plotly_chart(fig_3d, use_container_width=True)
                
                with tab3:
                    fig_sizes = plot_cluster_sizes(labels, algorithm="DBSCAN")
                    st.plotly_chart(fig_sizes, use_container_width=True)
                
                # Export results
                st.markdown("---")
                results_df = data_processed.copy()
                results_df['Cluster'] = labels
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="dbscan_results.csv",
                    mime="text/csv"
                )
            
            elif algorithm == "Hierarchical":
                st.subheader("📊 Hierarchical Clustering Results")
                
                # Run Hierarchical
                model, labels, metrics, linkage_matrix = hierarchical_clustering(
                    data_normalized,
                    n_clusters=n_clusters_hier,
                    linkage_method=linkage_method
                )
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Clusters", metrics['n_clusters'])
                with col2:
                    st.metric("Silhouette Score", f"{metrics['silhouette']:.3f}" if metrics['silhouette'] else "N/A")
                with col3:
                    st.metric("Davies-Bouldin", f"{metrics['davies_bouldin']:.3f}" if metrics['davies_bouldin'] else "N/A")
                
                # Visualizations
                tab1, tab2, tab3, tab4 = st.tabs(["Dendrogram", "2D Clusters", "3D Clusters", "Cluster Sizes"])
                
                with tab1:
                    fig_dend = plot_dendrogram(linkage_matrix)
                    st.plotly_chart(fig_dend, use_container_width=True)
                
                with tab2:
                    fig_2d = plot_clusters_2d(data_normalized, labels, algorithm="Hierarchical")
                    st.plotly_chart(fig_2d, use_container_width=True)
                
                with tab3:
                    fig_3d = plot_clusters_3d(data_normalized, labels, algorithm="Hierarchical")
                    st.plotly_chart(fig_3d, use_container_width=True)
                
                with tab4:
                    fig_sizes = plot_cluster_sizes(labels, algorithm="Hierarchical")
                    st.plotly_chart(fig_sizes, use_container_width=True)
                
                # Export results
                st.markdown("---")
                results_df = data_processed.copy()
                results_df['Cluster'] = labels
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="hierarchical_results.csv",
                    mime="text/csv"
                )
            
            elif algorithm == "Compare All":
                st.subheader("📊 Algorithm Comparison")
                
                # Run all algorithms
                with st.spinner('Running K-means...'):
                    _, kmeans_labels, kmeans_metrics = kmeans_clustering(
                        data_normalized,
                        n_clusters=n_clusters if not auto_k else find_optimal_k(data_normalized)['best_k']
                    )
                
                with st.spinner('Running DBSCAN...'):
                    _, dbscan_labels, dbscan_metrics = dbscan_clustering(
                        data_normalized,
                        eps=eps,
                        min_samples=min_samples
                    )
                
                with st.spinner('Running Hierarchical...'):
                    _, hier_labels, hier_metrics, _ = hierarchical_clustering(
                        data_normalized,
                        n_clusters=n_clusters_hier,
                        linkage_method=linkage_method
                    )
                
                # Compare metrics
                comparison = compare_algorithms(
                    data_normalized,
                    kmeans_labels,
                    dbscan_labels,
                    hier_labels
                )
                
                # Metrics comparison
                st.subheader("Performance Metrics")
                fig_comp = plot_metrics_comparison(comparison)
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Side-by-side visualizations
                st.subheader("Visual Comparison")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**K-means**")
                    fig_km = plot_clusters_2d(data_normalized, kmeans_labels, title="K-means", algorithm="")
                    st.plotly_chart(fig_km, use_container_width=True)
                    st.metric("Clusters", comparison['K-means']['n_clusters'])
                    if comparison['K-means'].get('silhouette'):
                        st.metric("Silhouette", f"{comparison['K-means']['silhouette']:.3f}")
                
                with col2:
                    st.markdown("**DBSCAN**")
                    fig_db = plot_clusters_2d(data_normalized, dbscan_labels, title="DBSCAN", algorithm="")
                    st.plotly_chart(fig_db, use_container_width=True)
                    st.metric("Clusters", comparison['DBSCAN']['n_clusters'])
                    st.metric("Noise Points", dbscan_metrics['n_noise'])
                
                with col3:
                    st.markdown("**Hierarchical**")
                    fig_hier = plot_clusters_2d(data_normalized, hier_labels, title="Hierarchical", algorithm="")
                    st.plotly_chart(fig_hier, use_container_width=True)
                    st.metric("Clusters", comparison['Hierarchical']['n_clusters'])
                    if comparison['Hierarchical'].get('silhouette'):
                        st.metric("Silhouette", f"{comparison['Hierarchical']['silhouette']:.3f}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🔬 Interactive Clustering Analysis Dashboard</p>
            <p>Developed by <strong>Álvaro Andrés De Lamo</strong></p>
            <p>University of Silesia - Cluster Analysis Course 🇵🇱</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
