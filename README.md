# 🔬 Interactive Clustering Analysis Dashboard

An interactive web application for performing and visualizing clustering analysis with multiple algorithms (K-means, DBSCAN, and Hierarchical Clustering). Built with Streamlit and Plotly for data scientists and analysts.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Developed by Álvaro Andrés De Lamo**  
University of Silesia - Cluster Analysis Course 🇵🇱

---

## ✨ Features

### 📊 **Multiple Clustering Algorithms**
- **K-means**: With automatic optimal k detection (elbow method + silhouette score)
- **DBSCAN**: Density-based clustering with noise detection
- **Hierarchical**: Agglomerative clustering with dendrogram visualization
- **Algorithm Comparison**: Side-by-side performance metrics and visualizations

### 🎨 **Interactive Visualizations**
- 2D and 3D cluster scatter plots (Plotly)
- Hierarchical clustering dendrograms
- Elbow method and silhouette score optimization curves
- Cluster size distribution charts
- Performance metrics comparison

### 🔧 **Data Preprocessing**
- Automatic handling of missing values (mean imputation)
- Categorical variable encoding (Label Encoding)
- Outlier detection and removal (IQR method with configurable threshold)
- Feature standardization (Z-score normalization)

### 📈 **Evaluation Metrics**
- Silhouette Score (cluster cohesion and separation)
- Davies-Bouldin Index (cluster compactness)
- Calinski-Harabasz Score (variance ratio)
- Inertia (within-cluster sum of squares for K-means)
- Noise detection (for DBSCAN)

### 💾 **Data Management**
- CSV file upload support
- Example dataset included (Absenteeism at Work)
- Export clustering results to CSV
- Data preview and statistical summary

---

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/aadl6-ua/cluster-analysis-dashboard.git
cd cluster-analysis-dashboard
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

### 1. **Load Data**
- **Option A**: Upload your own CSV file via the sidebar
- **Option B**: Use the included example dataset (Absenteeism at Work)

**Data Requirements**:
- CSV format (comma or semicolon separated)
- Numeric and/or categorical features
- First row should contain column headers

### 2. **Configure Preprocessing**
- Toggle outlier removal (IQR method)
- Adjust outlier threshold (1.0 - 3.0 IQR multipliers)

### 3. **Select Algorithm**
Choose from:
- **K-means**: Fast, spherical clusters
- **DBSCAN**: Density-based, handles noise
- **Hierarchical**: Tree-based, visualizes relationships
- **Compare All**: Run all three and compare

### 4. **Set Parameters**

**K-means**:
- Auto-detect optimal k (uses silhouette score)
- Or manually set number of clusters (2-10)

**DBSCAN**:
- Epsilon (ε): Maximum distance between points (0.1 - 2.0)
- Min samples: Minimum points to form a cluster (2-20)

**Hierarchical**:
- Number of clusters (2-10)
- Linkage method: ward, complete, average, single

### 5. **Run Analysis**
Click **"🚀 Run Analysis"** to:
- Preprocess the data
- Apply clustering algorithm(s)
- Generate visualizations
- Display evaluation metrics

### 6. **Explore Results**
- View 2D/3D cluster visualizations
- Analyze performance metrics
- Download results as CSV

---

## 📂 Project Structure

```
cluster-analysis-dashboard/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                 # Git ignore rules
├── utils/
│   ├── preprocessing.py       # Data loading and preprocessing
│   ├── clustering.py          # Clustering algorithms
│   └── visualization.py       # Plotly visualizations
├── data/
│   └── Absenteeism_at_work.csv  # Example dataset
├── docs/
│   └── screenshots/           # Documentation images
└── assets/
    └── logo.png               # App logo
```

---

## 🎯 Algorithm Details

### K-means Clustering
- **Type**: Partitioning
- **Best for**: Spherical, similarly-sized clusters
- **Parameters**: Number of clusters (k)
- **Optimization**: Elbow method + Silhouette score

### DBSCAN (Density-Based Spatial Clustering)
- **Type**: Density-based
- **Best for**: Arbitrary-shaped clusters, outlier detection
- **Parameters**: Epsilon (neighborhood radius), min_samples
- **Advantages**: Handles noise, doesn't require k

### Hierarchical Clustering
- **Type**: Agglomerative (bottom-up)
- **Best for**: Understanding data hierarchy
- **Parameters**: Number of clusters, linkage method
- **Visualization**: Dendrogram shows merge process

---

## 📊 Example Dataset

**Absenteeism at Work** dataset includes:
- Employee demographics (age, education, etc.)
- Work-related factors (service time, workload)
- Health indicators (BMI, diseases)
- Absenteeism hours

**Use case**: Identify patterns in employee absenteeism to improve workplace policies.

---

## 🌐 Deployment

### Deploy to Streamlit Cloud (Free)

1. **Push to GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Streamlit Cloud**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Sign in with GitHub
- Select your repository
- Set main file path: `app.py`
- Deploy!

Your app will be live at: `https://[your-username]-cluster-analysis-dashboard.streamlit.app`

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit 1.31+
- **Visualization**: Plotly 5.18, Matplotlib, Seaborn
- **ML Libraries**: scikit-learn 1.4, scipy 1.12
- **Data Processing**: pandas 2.2, numpy 1.26
- **Deployment**: Streamlit Cloud

---

## 📝 Features Roadmap

- [ ] Support for more algorithms (OPTICS, Mean Shift)
- [ ] PCA/t-SNE dimensionality reduction visualization
- [ ] Cluster profiling and interpretation
- [ ] Batch processing for multiple files
- [ ] Custom metric selection
- [ ] Export visualizations as PNG/PDF

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Álvaro Andrés De Lamo**
- GitHub: [@aadl6-ua](https://github.com/aadl6-ua)
- LinkedIn: [Álvaro Andrés De Lamo](https://www.linkedin.com/in/alvaro-andres-de-lamo-50149b358)
- Portfolio: [aadl6-ua.github.io](https://aadl6-ua.github.io)

---

## 🎓 Academic Context

This project was developed as part of the **Cluster Analysis** course at the **University of Silesia** (Poland) during an Erasmus+ exchange program (October 2024 - February 2025) as part of the Computer Engineering degree at Universidad de Alicante, Spain.

---

## 📸 Screenshots

### Main Dashboard
![Dashboard Overview](docs/screenshots/dashboard.png)

### K-means with Optimization
![K-means Analysis](docs/screenshots/kmeans.png)

### Algorithm Comparison
![Comparison View](docs/screenshots/comparison.png)

---

## 🙏 Acknowledgments

- University of Silesia for the Cluster Analysis course
- Streamlit team for the amazing framework
- scikit-learn contributors for robust ML algorithms

---

**⭐ If you find this project useful, please consider giving it a star!**
