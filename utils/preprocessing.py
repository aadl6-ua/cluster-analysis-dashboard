"""
Clustering utilities module - Data preprocessing functions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def load_data(file):
    """Load CSV file and perform initial cleaning"""
    try:
        # Try different separators
        df = pd.read_csv(file, sep=';')
        if len(df.columns) == 1:
            file.seek(0)
            df = pd.read_csv(file, sep=',')
        
        # Remove ID column if exists
        if 'ID' in df.columns:
            df = df.drop('ID', axis=1)
        
        return df, None
    except Exception as e:
        return None, str(e)


def preprocess_data(df, remove_outliers=True, outlier_threshold=1.5):
    """
    Preprocess data with imputation, encoding, normalization and outlier removal
    
    Parameters:
    - df: Input DataFrame
    - remove_outliers: Whether to remove outliers
    - outlier_threshold: IQR multiplier for outlier detection
    
    Returns:
    - df_normalized: Normalized DataFrame
    - df_processed: Processed DataFrame (before normalization)
    - preprocessing_info: Dictionary with preprocessing statistics
    """
    df_processed = df.copy()
    info = {
        'original_shape': df.shape,
        'missing_values': df.isnull().sum().sum(),
        'categorical_cols': [],
        'numeric_cols': [],
        'outliers_removed': 0
    }
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    info['numeric_cols'] = numeric_columns
    
    if len(numeric_columns) > 0:
        df_processed[numeric_columns] = imputer.fit_transform(df_processed[numeric_columns])
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
    info['categorical_cols'] = categorical_columns
    
    for col in categorical_columns:
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Remove outliers using IQR method
    if remove_outliers and len(numeric_columns) > 0:
        df_before_outliers = df_processed.copy()
        
        for col in numeric_columns:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            df_processed = df_processed[
                (df_processed[col] >= lower_bound) & 
                (df_processed[col] <= upper_bound)
            ]
        
        info['outliers_removed'] = len(df_before_outliers) - len(df_processed)
    
    # Normalization
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df_processed),
        columns=df_processed.columns,
        index=df_processed.index
    )
    
    info['final_shape'] = df_processed.shape
    
    return df_normalized, df_processed, info


def get_feature_summary(df):
    """Get summary statistics for features"""
    summary = {
        'numeric': {},
        'categorical': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'missing': df[col].isnull().sum()
        }
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        summary['categorical'][col] = {
            'unique': df[col].nunique(),
            'top': df[col].mode()[0] if len(df[col]) > 0 else None,
            'missing': df[col].isnull().sum()
        }
    
    return summary
