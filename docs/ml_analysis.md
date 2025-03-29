# Machine Learning Analysis Module

The Machine Learning Analysis module provides powerful algorithms for unsupervised and supervised learning tasks on biological data. This component enables researchers to discover patterns, classify samples, and build predictive models without requiring extensive programming knowledge.

## Key Features

1. **Clustering Analysis**: Group similar samples to discover natural patterns in data
2. **Predictive Modeling**: Build regression models to predict numeric outcomes
3. **Feature Importance**: Identify influential variables in prediction tasks
4. **Model Evaluation**: Assess model performance with standard metrics
5. **Visualization**: View clustering results and prediction accuracy

## Core Functions

### Clustering Analysis

#### perform_clustering(data, columns, method='K-Means', params=None, standardize=True)
Performs clustering analysis on data to group similar samples.

```python
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
```

#### Supported Clustering Methods

1. **K-Means Clustering**:
   - Fast and efficient for large datasets
   - Parameters:
     - `n_clusters`: Number of clusters to form
     - `random_state`: Seed for random initialization

2. **Hierarchical Clustering**:
   - Creates nested clusters in a hierarchical structure
   - Parameters:
     - `n_clusters`: Number of clusters to form
     - `linkage`: Linkage method ('ward', 'complete', 'average', 'single')

3. **DBSCAN (Density-Based Spatial Clustering)**:
   - Identifies clusters of arbitrary shape
   - Automatically detects outliers
   - Parameters:
     - `eps`: Maximum distance between samples in a cluster
     - `min_samples`: Minimum number of samples in a neighborhood

### Predictive Modeling

#### perform_simple_ml_prediction(data, target, predictors, model_type='Linear Regression', test_size=0.2, random_state=42)
Builds and evaluates a machine learning model for predicting outcomes.

```python
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
```

#### Supported Prediction Methods

1. **Linear Regression**:
   - Predicts continuous values
   - Provides interpretable coefficients
   - Fast training and prediction

2. **Random Forest**:
   - Ensemble method using multiple decision trees
   - Can capture non-linear relationships
   - Provides reliable feature importance

3. **Support Vector Machine (SVM)**:
   - Good for high-dimensional data
   - Effective with smaller datasets
   - Can handle non-linear relationships with kernels

## Usage Examples

### Clustering Gene Expression Data

```python
from components.ml_analysis import perform_clustering
import pandas as pd

# Example gene expression data
data = pd.DataFrame({
    'Gene1': [0.5, 0.7, 2.3, 2.5, 0.6, 2.1],
    'Gene2': [1.1, 1.3, 3.1, 3.0, 1.0, 3.2],
    'Gene3': [0.8, 0.9, 2.8, 2.7, 0.7, 2.9],
    'Sample': ['C1', 'C2', 'T1', 'T2', 'C3', 'T3']
})

# Perform k-means clustering
clustering_results = perform_clustering(
    data=data,
    columns=['Gene1', 'Gene2', 'Gene3'],
    method='K-Means',
    params={'n_clusters': 2},
    standardize=True
)

# Access clustering results
labels = clustering_results['labels']
centers = clustering_results['centers']
print(f"Cluster Labels: {labels}")
print(f"Cluster Centers:\n{centers}")
```

### Predicting Protein Binding Affinity

```python
from components.ml_analysis import perform_simple_ml_prediction
import pandas as pd

# Example protein binding data
data = pd.DataFrame({
    'Hydrophobicity': [0.5, 0.7, 0.3, 0.8, 0.6, 0.4],
    'Charge': [-1, -2, 0, -1, -3, 1],
    'Size': [142, 156, 120, 163, 178, 131],
    'Temperature': [25, 30, 20, 35, 25, 20],
    'BindingAffinity': [0.42, 0.57, 0.28, 0.63, 0.51, 0.35]
})

# Predict binding affinity
prediction_results = perform_simple_ml_prediction(
    data=data,
    target='BindingAffinity',
    predictors=['Hydrophobicity', 'Charge', 'Size', 'Temperature'],
    model_type='Random Forest',
    test_size=0.3,
    random_state=42
)

# Access prediction results
r2_score = prediction_results['r2_test']
feature_importance = prediction_results['feature_importance']
print(f"R² Score: {r2_score:.2f}")
print("Feature Importance:")
for feature, importance in zip(predictors, feature_importance):
    print(f"  {feature}: {importance:.3f}")
```

## Algorithm Selection Guide

### When to Use Each Clustering Method

| Method | Best For | Limitations |
|--------|----------|-------------|
| K-Means | Large datasets, spherical clusters | Requires number of clusters in advance, sensitive to outliers |
| Hierarchical | Visualizing cluster relationships, small-medium datasets | Computationally intensive for large datasets |
| DBSCAN | Finding clusters of arbitrary shape, automatic outlier detection | Sensitive to parameter selection, struggles with varying densities |

### When to Use Each Prediction Method

| Method | Best For | Limitations |
|--------|----------|-------------|
| Linear Regression | Simple relationships, interpretability, small datasets | Cannot capture non-linear relationships |
| Random Forest | Non-linear relationships, robust predictions, feature importance | Less interpretable, more computationally intensive |
| SVM | High-dimensional data, complex relationships | Parameter tuning can be difficult, slower on large datasets |

## Parameter Tuning Guide

### Clustering Parameters

**K-Means**:
- `n_clusters`: Start with domain knowledge (expected number of groups) or try a range of values and evaluate
- Determine optimal clusters using metrics like silhouette score or elbow method

**Hierarchical**:
- `linkage`: 'ward' often works well for gene expression data, 'average' for more general cases
- Visualize dendrograms to help determine appropriate cut points

**DBSCAN**:
- `eps`: Start with reasonable proximity expectation in your data's scale
- `min_samples`: Higher values create more significant clusters and more noise points

### Prediction Parameters

**General**:
- `test_size`: 0.2-0.3 is standard; use larger test sets for smaller datasets
- `random_state`: Fix this for reproducibility

**Model-Specific**:
- Random Forest: Consider adjusting `n_estimators` (number of trees) and `max_depth`
- SVM: Kernel selection ('linear', 'rbf', 'poly') significantly impacts performance

## Best Practices

1. **Preprocessing**:
   - Always standardize data for clustering and most prediction methods
   - Handle missing values appropriately before ML analysis
   - Remove or carefully handle outliers

2. **Validation**:
   - For clustering, validate results using domain knowledge
   - For prediction, use cross-validation when possible
   - Examine residuals and error distributions

3. **Interpretation**:
   - Don't rely solely on statistical metrics
   - Validate findings against biological knowledge
   - Consider biological plausibility of identified patterns

4. **Data Splitting**:
   - For small datasets (<100 samples), consider using cross-validation instead of train/test split
   - Ensure train and test sets represent the same distribution

## Performance Considerations

- Large datasets (>10,000 rows) may require:
  - Sampling for initial exploration
  - Algorithm selection based on computational efficiency
  - Incremental learning approaches for very large datasets

## Dependencies

- scikit-learn: For core machine learning algorithms
- numpy: For numerical operations
- pandas: For data handling
- scipy: For hierarchical clustering and distance calculations

## Extension Points

The ML Analysis module can be extended by:
1. Adding new clustering algorithms (e.g., Spectral Clustering, GMM)
2. Implementing classification models for categorical outcomes
3. Adding advanced model validation techniques
4. Implementing feature selection methods