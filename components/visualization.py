import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def create_scatter_plot(data, x, y, color=None, title=None, size=None):
    """
    Create an interactive scatter plot.
    
    Args:
        data: DataFrame with data to plot
        x: Column name for x-axis
        y: Column name for y-axis
        color: Optional column name for color encoding
        title: Optional plot title
        size: Optional column name for point size
    
    Returns:
        plotly.graph_objects.Figure: Scatter plot
    """
    if not title:
        title = f"{y} vs {x}"
        if color:
            title += f" (grouped by {color})"
    
    fig = px.scatter(
        data, 
        x=x, 
        y=y, 
        color=color,
        size=size,
        title=title,
        hover_data=data.columns,
        template="plotly_white"
    )
    
    fig.update_layout(
        xaxis_title=x,
        yaxis_title=y,
        legend_title_text=color if color else "",
        height=600
    )
    
    # Add trendline if no color grouping
    if not color and len(data) > 2:
        df = data[[x, y]].dropna()
        if len(df) > 2:  # Need at least 3 points for regression
            fig.add_trace(
                go.Scatter(
                    x=df[x],
                    y=df[y].values,
                    mode='markers',
                    marker=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                    hoverinfo='none'
                )
            )
            
            # Add trendline with Plotly
            fig.update_traces(
                selector=dict(mode='markers'),
                trendline='ols',
                trendline_color_override='red'
            )
    
    return fig

def create_box_plot(data, numeric_column, group_column, groups=None):
    """
    Create a box plot for a numeric variable grouped by a categorical variable.
    
    Args:
        data: DataFrame with data to plot
        numeric_column: Column name for the numeric variable
        group_column: Column name for the grouping variable
        groups: Optional list of groups to include (subset of group_column values)
    
    Returns:
        plotly.graph_objects.Figure: Box plot
    """
    # Filter by selected groups if provided
    if groups:
        plot_data = data[data[group_column].isin(groups)].copy()
    else:
        plot_data = data.copy()
    
    fig = px.box(
        plot_data,
        x=group_column,
        y=numeric_column,
        color=group_column,
        title=f"{numeric_column} by {group_column}",
        template="plotly_white",
        points="outliers"  # Show outliers only
    )
    
    fig.update_layout(
        xaxis_title=group_column,
        yaxis_title=numeric_column,
        height=600,
        showlegend=False
    )
    
    return fig

def create_histogram(data, column=None, bins=20, color=None, y=None, title=None):
    """
    Create a histogram for the specified column.
    
    Args:
        data: DataFrame with data to plot
        column: Column name to plot (x-axis)
        bins: Number of bins for histogram
        color: Optional column name for color grouping
        y: Optional column name for y-axis (for bar charts)
        title: Optional plot title
    
    Returns:
        plotly.graph_objects.Figure: Histogram
    """
    if y is not None:
        # Bar chart mode (when providing both x and y)
        if not title:
            title = f"{y} by {column}"
        
        fig = px.bar(
            data,
            x=column,
            y=y,
            color=color,
            title=title,
            template="plotly_white"
        )
        
        fig.update_layout(
            xaxis_title=column,
            yaxis_title=y,
            height=600
        )
    else:
        # Standard histogram mode
        if not title:
            title = f"Distribution of {column}"
            if color:
                title += f" (grouped by {color})"
        
        fig = px.histogram(
            data,
            x=column,
            color=color,
            nbins=bins,
            title=title,
            template="plotly_white",
            barmode='overlay',
            opacity=0.7,
            marginal="box"  # Add box plot on the margin
        )
        
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Count",
            height=600
        )
    
    return fig

def create_heatmap(corr_matrix, title="Correlation Heatmap"):
    """
    Create a heatmap visualization of a correlation matrix.
    
    Args:
        corr_matrix: Correlation matrix (DataFrame)
        title: Plot title
    
    Returns:
        plotly.graph_objects.Figure: Heatmap
    """
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title=title,
        range_color=[-1, 1],
        template="plotly_white"
    )
    
    fig.update_layout(
        height=600,
        xaxis_title="",
        yaxis_title=""
    )
    
    return fig

def create_pca_plot(data, columns, n_components=2, color=None, standardize=True, title=None):
    """
    Create a PCA plot for the specified columns.
    
    Args:
        data: DataFrame with data to plot
        columns: List of column names to include in PCA
        n_components: Number of PCA components to calculate
        color: Optional column name for color encoding
        standardize: Whether to standardize data before PCA
        title: Optional plot title
    
    Returns:
        tuple: (plotly.graph_objects.Figure, np.array, pd.DataFrame) - The figure, explained variance ratios, and loadings
    """
    # Extract numeric data for PCA
    X = data[columns].copy()
    
    # Skip rows with any NaN values in the selected columns
    X = X.dropna(subset=columns)
    
    # Store indices for later joining with color column
    indices = X.index
    
    # Standardize the data if requested
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)
    
    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(
        data=pca_result,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=indices
    )
    
    # Add color column if provided
    if color and color in data.columns:
        pca_df[color] = data.loc[indices, color]
    
    # Calculate explained variance and prepare for plotting
    explained_variance = pca.explained_variance_ratio_
    
    # Create loadings dataframe
    loadings = pd.DataFrame(
        data=pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=columns
    )
    
    # Create plot title if not provided
    if not title:
        var_sum = sum(explained_variance[:2]) * 100
        title = f"PCA Plot (PC1 vs PC2, {var_sum:.1f}% variance explained)"
        if color:
            title += f" (colored by {color})"
    
    # Create the scatter plot for the first two components
    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color=color,
        title=title,
        labels={
            'PC1': f'PC1 ({explained_variance[0]*100:.1f}%)',
            'PC2': f'PC2 ({explained_variance[1]*100:.1f}%)'
        },
        template="plotly_white"
    )
    
    # Add loadings as vectors (for the first two components)
    for i, feature in enumerate(columns):
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loadings.iloc[i, 0] * 0.5,  # Scale down the arrows
            y1=loadings.iloc[i, 1] * 0.5,
            line=dict(color='red', width=1),
            opacity=0.8
        )
        
        # Add text labels for the loadings
        fig.add_annotation(
            x=loadings.iloc[i, 0] * 0.5,
            y=loadings.iloc[i, 1] * 0.5,
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
            showarrow=False
        )
    
    fig.update_layout(
        height=600,
        legend_title=color if color else ""
    )
    
    return fig, explained_variance, loadings
