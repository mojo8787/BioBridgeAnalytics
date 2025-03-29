# Data Visualization Component

The Visualization component provides powerful, interactive data visualization capabilities for biological datasets. It leverages Plotly to create publication-quality, interactive plots that can be customized, exported, and embedded in reports.

## Visualization Types

1. **Scatter Plots**: Visualize relationships between two variables with optional coloring
2. **Box Plots**: Compare distributions across categorical groups
3. **Histograms**: Examine value distributions and frequencies
4. **Heatmaps**: Visualize correlation matrices and other grid data
5. **PCA Plots**: Reveal patterns in high-dimensional data through dimensionality reduction

## Core Functions

### create_scatter_plot(data, x, y, color=None, title=None, size=None)
Creates an interactive scatter plot with optional color encoding.

```python
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
```

### create_box_plot(data, numeric_column, group_column, groups=None)
Creates a box plot for comparing a numeric variable across groups.

```python
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
```

### create_histogram(data, column=None, bins=20, color=None, y=None, title=None)
Creates a histogram for examining value distributions.

```python
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
```

### create_heatmap(corr_matrix, title="Correlation Heatmap")
Creates a heatmap visualization of a correlation matrix.

```python
def create_heatmap(corr_matrix, title="Correlation Heatmap"):
    """
    Create a heatmap visualization of a correlation matrix.
    
    Args:
        corr_matrix: Correlation matrix (DataFrame)
        title: Plot title
    
    Returns:
        plotly.graph_objects.Figure: Heatmap
    """
```

### create_pca_plot(data, columns, n_components=2, color=None, standardize=True, title=None)
Creates a PCA plot for visualizing high-dimensional data in a lower-dimensional space.

```python
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
```

## Plot Customization

All visualization functions support customization through:
- Color schemes and palettes
- Axis labels and titles
- Legend positioning
- Marker styles and sizes
- Hover information

## Interactive Features

Plots created by the visualization component include:
- Zoom and pan controls
- Hover tooltips with data points
- Click events for selection
- Export options (PNG, SVG, PDF)
- Full-screen mode

## Usage Examples

### Creating a Scatter Plot

```python
from components.visualization import create_scatter_plot
import pandas as pd

# Example data
data = pd.DataFrame({
    'Gene_Expression': [2.5, 3.1, 4.2, 3.8, 2.9, 3.5],
    'Protein_Level': [1.8, 2.2, 3.3, 2.9, 2.1, 2.7],
    'Sample_Type': ['Control', 'Control', 'Treatment', 'Treatment', 'Control', 'Treatment']
})

# Create scatter plot with colored points by sample type
fig = create_scatter_plot(
    data=data,
    x='Gene_Expression',
    y='Protein_Level',
    color='Sample_Type',
    title='Gene Expression vs. Protein Levels'
)

# Display the plot (in Streamlit)
st.plotly_chart(fig, use_container_width=True)
```

### Creating a Correlation Heatmap

```python
from components.visualization import create_heatmap
import pandas as pd
import numpy as np

# Example correlation matrix
data = pd.DataFrame(np.random.rand(5, 5), 
                    columns=['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5'],
                    index=['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5'])
corr_matrix = data.corr()

# Create heatmap
fig = create_heatmap(
    corr_matrix=corr_matrix,
    title='Gene Correlation Heatmap'
)

# Display the plot (in Streamlit)
st.plotly_chart(fig, use_container_width=True)
```

### Creating a PCA Plot

```python
from components.visualization import create_pca_plot
import pandas as pd

# Example data with multiple numeric columns
data = pd.DataFrame({
    'Gene1': [1.2, 2.3, 3.4, 2.5, 1.8, 3.1],
    'Gene2': [2.1, 3.2, 4.3, 3.4, 2.7, 4.0],
    'Gene3': [0.9, 1.8, 2.7, 1.6, 1.1, 2.4],
    'Gene4': [3.1, 4.2, 5.3, 4.4, 3.7, 5.0],
    'Sample_Type': ['Control', 'Control', 'Treatment', 'Treatment', 'Control', 'Treatment']
})

# Create PCA plot
fig, explained_variance, loadings = create_pca_plot(
    data=data,
    columns=['Gene1', 'Gene2', 'Gene3', 'Gene4'],
    color='Sample_Type',
    title='PCA Analysis of Gene Expression'
)

# Display the plot (in Streamlit)
st.plotly_chart(fig, use_container_width=True)

# Display explained variance
st.write(f"Explained variance: {explained_variance[0]:.2f}%, {explained_variance[1]:.2f}%")
```

## Exporting Visualizations

Plots can be exported from the UI in multiple formats:
- PNG: For images in reports and presentations
- SVG: For vector graphics and publication
- HTML: For interactive web embedding
- PDF: For publication-ready figures

## Colors and Styling

The visualization component uses consistent color schemes that:
- Are color-blind friendly
- Provide good contrast
- Follow scientific publication standards
- Scale appropriately for categorical variables

## Best Practices

When using the visualization component:
1. Choose appropriate plot types for the data and question
2. Normalize or transform data as needed before visualization
3. Use color encoding sparingly and meaningfully
4. Add clear titles and axis labels
5. Consider adding annotations for important features

## Dependencies

- plotly: For creating interactive visualizations
- numpy: For numerical operations
- pandas: For data manipulation
- scikit-learn: For PCA analysis

## Extension Points

The visualization component can be extended by:
1. Adding new plot types (network graphs, 3D plots, etc.)
2. Implementing additional dimensionality reduction methods (t-SNE, UMAP)
3. Creating domain-specific visualizations (sequence visualizations, pathway diagrams)
4. Adding animation capabilities for time-series data