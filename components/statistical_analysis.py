import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st

def perform_descriptive_stats(data, columns):
    """
    Calculate descriptive statistics for specified columns.
    
    Args:
        data: DataFrame with data to analyze
        columns: List of column names to analyze
    
    Returns:
        DataFrame: Descriptive statistics results
    """
    if not all(col in data.columns for col in columns):
        st.error("Some selected columns are not in the dataset")
        return None
    
    # Filter to only the selected columns
    selected_data = data[columns].copy()
    
    # Calculate statistics
    desc_stats = selected_data.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
    
    # Add additional statistics
    additional_stats = pd.DataFrame(index=desc_stats.index)
    
    # Calculate skewness
    additional_stats['skewness'] = selected_data.skew()
    
    # Calculate kurtosis
    additional_stats['kurtosis'] = selected_data.kurt()
    
    # Calculate coefficient of variation (CV)
    additional_stats['cv'] = selected_data.std() / selected_data.mean() * 100
    
    # Calculate missing values
    additional_stats['missing'] = selected_data.isna().sum()
    additional_stats['missing_pct'] = selected_data.isna().mean() * 100
    
    # Combine with the standard describe() output
    result = pd.concat([desc_stats, additional_stats], axis=1)
    
    # Round all float values for display
    result = result.round(4)
    
    return result

def perform_correlation_analysis(data, columns, method='pearson'):
    """
    Calculate correlation matrix and p-values for specified columns.
    
    Args:
        data: DataFrame with data to analyze
        columns: List of column names to analyze
        method: Correlation method ('pearson', 'spearman', or 'kendall')
    
    Returns:
        tuple: (correlation_matrix, p_values) as DataFrames
    """
    if not all(col in data.columns for col in columns):
        st.error("Some selected columns are not in the dataset")
        return None, None
    
    # Filter to only the selected columns
    selected_data = data[columns].copy()
    
    # Calculate correlation matrix
    corr_matrix = selected_data.corr(method=method)
    
    # Calculate p-values
    p_values = pd.DataFrame(np.zeros((len(columns), len(columns))), index=columns, columns=columns)
    
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i != j:  # Skip self-correlations
                if method == 'pearson':
                    corr, p = stats.pearsonr(selected_data[col1].dropna(), selected_data[col2].dropna())
                elif method == 'spearman':
                    corr, p = stats.spearmanr(selected_data[col1].dropna(), selected_data[col2].dropna())
                elif method == 'kendall':
                    corr, p = stats.kendalltau(selected_data[col1].dropna(), selected_data[col2].dropna())
                else:
                    st.error(f"Unsupported correlation method: {method}")
                    return None, None
                
                p_values.loc[col1, col2] = p
                p_values.loc[col2, col1] = p  # Symmetric matrix
            else:
                p_values.loc[col1, col2] = 1.0  # p-value of 1 for self-correlation
    
    return corr_matrix, p_values

def perform_ttest(data, numeric_column, group_column, group1, group2, equal_var=True):
    """
    Perform t-test comparing a numeric variable between two groups.
    
    Args:
        data: DataFrame with data to analyze
        numeric_column: Column name for the numeric variable
        group_column: Column name for the grouping variable
        group1: First group value
        group2: Second group value
        equal_var: Whether to assume equal variance (default: True)
    
    Returns:
        dict: T-test results
    """
    if not (numeric_column in data.columns and group_column in data.columns):
        st.error("Specified columns not found in dataset")
        return None
    
    # Extract the two groups
    group1_data = data[data[group_column] == group1][numeric_column].dropna()
    group2_data = data[data[group_column] == group2][numeric_column].dropna()
    
    if len(group1_data) < 2 or len(group2_data) < 2:
        st.error(f"Not enough data points in one or both groups")
        return None
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)
    
    # Calculate additional statistics
    mean1 = group1_data.mean()
    mean2 = group2_data.mean()
    std1 = group1_data.std()
    std2 = group2_data.std()
    n1 = len(group1_data)
    n2 = len(group2_data)
    mean_diff = mean1 - mean2
    
    # Calculate Cohen's d effect size
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = mean_diff / pooled_std if pooled_std != 0 else np.nan
    
    # Return results
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean1': mean1,
        'mean2': mean2,
        'std1': std1,
        'std2': std2,
        'n1': n1,
        'n2': n2,
        'mean_diff': mean_diff,
        'cohens_d': cohens_d
    }

def perform_anova(data, numeric_column, group_column):
    """
    Perform one-way ANOVA test comparing a numeric variable across groups.
    
    Args:
        data: DataFrame with data to analyze
        numeric_column: Column name for the numeric variable
        group_column: Column name for the grouping variable
    
    Returns:
        dict: ANOVA results
    """
    if not (numeric_column in data.columns and group_column in data.columns):
        st.error("Specified columns not found in dataset")
        return None
    
    # Group data by the grouping variable
    groups = data.groupby(group_column)[numeric_column].apply(lambda x: x.dropna().values).to_dict()
    
    # Filter out empty groups
    groups = {k: v for k, v in groups.items() if len(v) > 0}
    
    if len(groups) < 2:
        st.error("At least two non-empty groups are required for ANOVA")
        return None
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*list(groups.values()))
    
    # Calculate group statistics
    group_stats = data.groupby(group_column)[numeric_column].agg(['mean', 'std', 'count']).to_dict()
    
    # Return results
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'group_stats': group_stats,
        'total_observations': sum(len(v) for v in groups.values())
    }
