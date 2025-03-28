import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st

def process_data(data):
    """
    Process the loaded data: clean, transform, and prepare for analysis.
    
    Args:
        data: Input data (DataFrame or other format)
        
    Returns:
        DataFrame: Processed data ready for analysis
    """
    if not isinstance(data, pd.DataFrame):
        st.error("Data processing requires a pandas DataFrame")
        return None
    
    # Make a copy to avoid modifying the original
    processed_data = data.copy()
    
    # Handle column names - strip whitespace and replace spaces with underscores
    processed_data.columns = [col.strip().replace(' ', '_') for col in processed_data.columns]
    
    # Try to convert columns to appropriate data types
    for col in processed_data.columns:
        # Skip if the column is already non-object type
        if processed_data[col].dtype != 'object':
            continue
        
        # Try to convert to numeric
        try:
            # Convert strings like '1,234' to '1234' first
            if processed_data[col].dtype == 'object':
                # Check if the column has comma-formatted numbers
                sample = processed_data[col].dropna().iloc[:5] if len(processed_data[col].dropna()) >= 5 else processed_data[col].dropna()
                if all(isinstance(x, str) and ',' in x and sum(c.isdigit() for c in x) > 0 for x in sample):
                    processed_data[col] = processed_data[col].str.replace(',', '')
            
            numeric_col = pd.to_numeric(processed_data[col], errors='coerce')
            
            # Only convert if at least 80% of values converted successfully
            if numeric_col.notna().sum() >= 0.8 * len(processed_data):
                processed_data[col] = numeric_col
        except Exception:
            # Keep as is if conversion fails
            pass
    
    # Check for and fix column names that might cause problems
    problematic_chars = ['+', '-', '*', '/', ':', ';', '!', '@', '#', '$', '%', '^', '&', '(', ')', ' ']
    for char in problematic_chars:
        processed_data.columns = [col.replace(char, '_') for col in processed_data.columns]
    
    # Remove rows with all NaN values
    processed_data = processed_data.dropna(how='all')
    
    # If there are duplicate column names, make them unique
    if len(processed_data.columns) != len(set(processed_data.columns)):
        processed_data.columns = pd.Series(processed_data.columns).map(
            lambda x: f"{x}_{list(processed_data.columns).count(x)}" if list(processed_data.columns).count(x) > 1 else x
        )
    
    return processed_data

def get_column_types(data):
    """
    Identify and categorize column types in the dataset.
    
    Args:
        data: DataFrame to analyze
        
    Returns:
        dict: Dictionary with column types and metadata
    """
    if not isinstance(data, pd.DataFrame):
        st.error("Column type identification requires a pandas DataFrame")
        return {}
    
    column_types = {}
    
    for col in data.columns:
        # Count missing values
        missing_count = data[col].isna().sum()
        missing_percentage = (missing_count / len(data)) * 100
        
        if pd.api.types.is_numeric_dtype(data[col]):
            # Numeric column
            column_types[col] = {
                'type': 'numeric',
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'min': data[col].min() if not data[col].empty else None,
                'max': data[col].max() if not data[col].empty else None,
                'mean': data[col].mean() if not data[col].empty else None,
                'median': data[col].median() if not data[col].empty else None
            }
        elif pd.api.types.is_datetime64_dtype(data[col]):
            # Datetime column
            column_types[col] = {
                'type': 'datetime',
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'min': data[col].min() if not data[col].empty else None,
                'max': data[col].max() if not data[col].empty else None
            }
        else:
            # Categorical or text column
            unique_count = data[col].nunique()
            total_count = len(data[col].dropna())
            
            # If less than 20% unique values or less than 10 unique values, consider categorical
            if (unique_count / total_count < 0.2 or unique_count < 10) and total_count > 0:
                column_types[col] = {
                    'type': 'categorical',
                    'missing_count': missing_count,
                    'missing_percentage': missing_percentage,
                    'unique_values': unique_count,
                    'most_common': data[col].value_counts().index[0] if not data[col].empty else None
                }
            else:
                column_types[col] = {
                    'type': 'text',
                    'missing_count': missing_count,
                    'missing_percentage': missing_percentage,
                    'unique_values': unique_count,
                    'avg_length': data[col].astype(str).map(len).mean() if not data[col].empty else None
                }
    
    return column_types

def identify_potential_correlations(data, threshold=0.7):
    """
    Identify potentially interesting correlations in the dataset.
    
    Args:
        data: DataFrame to analyze
        threshold: Correlation coefficient threshold (absolute value)
        
    Returns:
        list: List of tuples with correlated column pairs and correlation values
    """
    if not isinstance(data, pd.DataFrame):
        st.error("Correlation identification requires a pandas DataFrame")
        return []
    
    # Get numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 2:
        return []
    
    # Calculate correlation matrix
    corr_matrix = data[numeric_cols].corr(method='pearson')
    
    # Find pairs with correlation above threshold (ignoring self-correlations)
    high_corr_pairs = []
    
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            corr_value = corr_matrix.iloc[i, j]
            if not pd.isna(corr_value) and abs(corr_value) >= threshold:
                high_corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_value))
    
    # Sort by absolute correlation value (descending)
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Format as (col1, col2), corr_value for return
    return [(pair[0:2], pair[2]) for pair in high_corr_pairs]
