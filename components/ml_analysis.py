import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster

def perform_clustering(data, columns, method='K-Means', params=None, standardize=True):
    """
    Perform clustering analysis on the data.
    
    Args:
        data: DataFrame with data to analyze
        columns: List of column names to use for clustering
        method: Clustering method ('K-Means', 'Hierarchical', 'DBSCAN')
        params: Parameters for the clustering algorithm
        standardize: Whether to standardize data before clustering
        
    Returns:
        dict: Clustering results
    """
    if not all(col in data.columns for col in columns):
        st.error("Some selected columns are not in the dataset")
        return None
    
    # Extract data for clustering
    X = data[columns].copy().dropna()
    
    # Store indices for later joining with cluster labels
    indices = X.index
    
    # Standardize the data if requested
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    # Default parameters if none provided
    if params is None:
        params = {}
    
    # Perform clustering based on the selected method
    if method == 'K-Means':
        n_clusters = params.get('n_clusters', 3)
        
        # Create and fit the model
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        
        # Extract cluster centers (transform back if standardized)
        if standardize:
            centers = scaler.inverse_transform(model.cluster_centers_)
        else:
            centers = model.cluster_centers_
        
        # Create DataFrame with cluster centers
        center_df = pd.DataFrame(centers, columns=columns)
        center_df.index.name = 'Cluster'
        
        # Calculate silhouette score if more than one cluster
        if n_clusters > 1:
            from sklearn.metrics import silhouette_score
            try:
                silhouette = silhouette_score(X_scaled, labels)
            except:
                silhouette = np.nan
        else:
            silhouette = np.nan
        
        # Prepare results
        results = {
            'labels': pd.Series(labels, index=indices),
            'centers': center_df,
            'inertia': model.inertia_,
            'silhouette': silhouette,
            'n_clusters': n_clusters
        }
    
    elif method == 'Hierarchical':
        n_clusters = params.get('n_clusters', 3)
        linkage_method = params.get('linkage', 'ward')
        
        # Compute linkage matrix
        Z = linkage(X_scaled, method=linkage_method)
        
        # Extract cluster labels
        labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # 0-based indexing
        
        # Calculate cluster centers
        centers = np.zeros((n_clusters, X.shape[1]))
        for i in range(n_clusters):
            cluster_points = X_scaled[labels == i]
            if len(cluster_points) > 0:
                centers[i] = cluster_points.mean(axis=0)
        
        # Transform back if standardized
        if standardize:
            centers = scaler.inverse_transform(centers)
        
        # Create DataFrame with cluster centers
        center_df = pd.DataFrame(centers, columns=columns)
        center_df.index.name = 'Cluster'
        
        # Calculate silhouette score if more than one cluster
        if n_clusters > 1 and len(set(labels)) > 1:
            from sklearn.metrics import silhouette_score
            try:
                silhouette = silhouette_score(X_scaled, labels)
            except:
                silhouette = np.nan
        else:
            silhouette = np.nan
        
        # Prepare results
        results = {
            'labels': pd.Series(labels, index=indices),
            'centers': center_df,
            'silhouette': silhouette,
            'n_clusters': n_clusters,
            'linkage': linkage_method
        }
    
    elif method == 'DBSCAN':
        eps = params.get('eps', 0.5)
        min_samples = params.get('min_samples', 5)
        
        # Create and fit the model
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
        
        # Count the number of clusters (excluding noise points labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Calculate cluster centers (excluding noise points)
        centers = np.zeros((n_clusters, X.shape[1]))
        valid_clusters = [i for i in range(n_clusters)]
        
        for i, cluster_id in enumerate(sorted(set(labels))):
            if cluster_id != -1:  # Skip noise points
                cluster_points = X_scaled[labels == cluster_id]
                centers[i] = cluster_points.mean(axis=0)
        
        # Transform back if standardized
        if standardize:
            centers = scaler.inverse_transform(centers)
        
        # Create DataFrame with cluster centers
        center_df = pd.DataFrame(centers, columns=columns)
        center_df.index = valid_clusters
        center_df.index.name = 'Cluster'
        
        # Calculate silhouette score if more than one cluster and no all points are noise
        if n_clusters > 1 and -1 not in labels:
            from sklearn.metrics import silhouette_score
            try:
                silhouette = silhouette_score(X_scaled, labels)
            except:
                silhouette = np.nan
        else:
            silhouette = np.nan
        
        # Count noise points
        noise_count = sum(1 for l in labels if l == -1)
        
        # Prepare results
        results = {
            'labels': pd.Series(labels, index=indices),
            'centers': center_df,
            'silhouette': silhouette,
            'n_clusters': n_clusters,
            'eps': eps,
            'min_samples': min_samples,
            'noise_count': noise_count,
            'noise_percent': noise_count / len(labels) * 100 if len(labels) > 0 else 0
        }
    
    else:
        st.error(f"Unsupported clustering method: {method}")
        return None
    
    return results

def perform_simple_ml_prediction(data, target, predictors, model_type='Linear Regression', test_size=0.2, random_state=42):
    """
    Perform simple machine learning prediction on the data.
    
    Args:
        data: DataFrame with data to analyze
        target: Target variable column name
        predictors: List of predictor column names
        model_type: Type of prediction model
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        dict: Prediction results
    """
    # Check if target and predictors are in the dataset
    if target not in data.columns:
        st.error(f"Target variable '{target}' not found in dataset")
        return None
    
    if not all(col in data.columns for col in predictors):
        st.error("Some predictor variables are not in the dataset")
        return None
    
    # Extract features and target
    X = data[predictors].copy()
    y = data[target].copy()
    
    # Remove rows with any missing values
    valid_indices = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]
    
    if len(X) < 10:
        st.error("Not enough data points after removing missing values")
        return None
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train the model based on the selected type
    if model_type == 'Linear Regression':
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Get feature importance (coefficients)
        feature_importance = model.coef_
        
    elif model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        model.fit(X_train, y_train)
        
        # Get feature importance
        feature_importance = model.feature_importances_
        
    elif model_type == 'Support Vector Machine':
        model = SVR(kernel='rbf')
        model.fit(X_train, y_train)
        
        # SVR doesn't have feature importance, use permutation importance
        feature_importance = np.zeros(len(predictors))
        
    else:
        st.error(f"Unsupported model type: {model_type}")
        return None
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Return results
    return {
        'model': model,
        'model_type': model_type,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'feature_importance': feature_importance,
        'y_test': y_test,
        'y_pred': y_pred,
        'predictors': predictors
    }
