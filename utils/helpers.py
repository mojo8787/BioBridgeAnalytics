import pandas as pd
import numpy as np
import streamlit as st
import math

def format_bytes(size):
    """
    Format file size in bytes to human-readable format.
    
    Args:
        size: Size in bytes
        
    Returns:
        str: Human-readable size (e.g., "1.23 MB")
    """
    size_names = ["B", "KB", "MB", "GB", "TB"]
    if size == 0:
        return "0 B"
    
    i = int(math.floor(math.log(size, 1024)))
    p = math.pow(1024, i)
    s = round(size / p, 2)
    
    return f"{s} {size_names[i]}"

def get_data_summary(data):
    """
    Get a summary of the dataset.
    
    Args:
        data: DataFrame to summarize
        
    Returns:
        dict: Dictionary with summary statistics
    """
    if not isinstance(data, pd.DataFrame):
        return {
            "rows": 0,
            "cols": 0,
            "numeric_cols": 0,
            "categorical_cols": 0,
            "missing_values": 0
        }
    
    # Count numeric and categorical columns
    numeric_cols = 0
    categorical_cols = 0
    
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            numeric_cols += 1
        else:
            categorical_cols += 1
    
    # Count missing values
    missing_values = data.isna().sum().sum()
    
    return {
        "rows": len(data),
        "cols": len(data.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing_values": missing_values,
        "missing_percentage": (missing_values / (len(data) * len(data.columns))) * 100 if len(data) * len(data.columns) > 0 else 0
    }

def truncate_text(text, max_length=50):
    """
    Truncate text to specified length and add ellipsis if needed.
    
    Args:
        text: String to truncate
        max_length: Maximum length before truncation
        
    Returns:
        str: Truncated text
    """
    if isinstance(text, str) and len(text) > max_length:
        return text[:max_length] + "..."
    return text

def infer_column_type(series):
    """
    Infer the data type of a pandas Series.
    
    Args:
        series: Pandas Series to analyze
        
    Returns:
        str: Inferred data type ("numeric", "categorical", "datetime", "text")
    """
    if pd.api.types.is_numeric_dtype(series):
        # Check if integer-like with few unique values (potential categorical)
        if series.dropna().apply(lambda x: x.is_integer() if hasattr(x, 'is_integer') else True).all():
            unique_vals = series.nunique()
            if unique_vals <= 10 and unique_vals / len(series) < 0.05:
                return "categorical"
        return "numeric"
    
    elif pd.api.types.is_datetime64_dtype(series):
        return "datetime"
    
    else:
        # Check if categorical (few unique values)
        unique_vals = series.nunique()
        if unique_vals <= 20 or (unique_vals / len(series) < 0.1 and len(series) > 100):
            return "categorical"
        else:
            return "text"

def safe_convert_to_numeric(series):
    """
    Safely convert a series to numeric, handling errors gracefully.
    
    Args:
        series: Pandas Series to convert
        
    Returns:
        Series: Converted series or original if conversion failed
    """
    try:
        # Remove commas and other formatting characters
        if series.dtype == 'object':
            series = series.str.replace(',', '').str.replace('$', '').str.replace('%', '')
        
        # Convert to numeric
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        # Only return the numeric series if most values converted successfully
        if numeric_series.notna().sum() >= 0.5 * len(series):
            return numeric_series
    except:
        pass
    
    # Return original series if conversion failed
    return series
