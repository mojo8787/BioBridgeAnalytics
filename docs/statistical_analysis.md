# Statistical Analysis Module

The Statistical Analysis module provides essential statistical tools for exploring, summarizing, and analyzing biological data. This component enables researchers to gain insights into data distributions, identify correlations, and test hypotheses without requiring extensive statistical programming.

## Key Features

1. **Descriptive Statistics**: Calculate summary statistics for numerical variables
2. **Correlation Analysis**: Analyze relationships between variables using various methods
3. **Hypothesis Testing**: Perform t-tests and ANOVA for comparing groups
4. **Result Visualization**: Present statistical results in clear tables and visualizations
5. **Interpretation Assistance**: Provide statistical interpretation guidelines

## Core Functions

### Descriptive Statistics

#### perform_descriptive_stats(data, columns)
Calculates descriptive statistics for specified columns.

```python
def perform_descriptive_stats(data, columns):
    """
    Calculate descriptive statistics for specified columns.
    
    Args:
        data: DataFrame with data to analyze
        columns: List of column names to analyze
    
    Returns:
        DataFrame: Descriptive statistics results
    """
```

**Computed Statistics:**
- Count
- Mean
- Standard Deviation
- Minimum
- 25th Percentile
- Median (50th Percentile)
- 75th Percentile
- Maximum
- Skewness
- Kurtosis

### Correlation Analysis

#### perform_correlation_analysis(data, columns, method='pearson')
Calculates correlation matrix and p-values for specified columns.

```python
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
```

**Supported Correlation Methods:**
- **Pearson**: Measures linear correlation between variables
- **Spearman**: Assesses monotonic relationships (rank correlation)
- **Kendall**: Evaluates concordance between observations (rank correlation)

### Hypothesis Testing

#### perform_ttest(data, numeric_column, group_column, group1, group2, equal_var=True)
Performs t-test comparing a numeric variable between two groups.

```python
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
```

**T-test Results Include:**
- t-statistic
- p-value
- Degrees of freedom
- Mean of each group
- Standard deviation of each group
- 95% confidence interval
- Cohen's d effect size

#### perform_anova(data, numeric_column, group_column)
Performs one-way ANOVA test comparing a numeric variable across groups.

```python
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
```

**ANOVA Results Include:**
- F-statistic
- p-value
- Degrees of freedom
- Group means
- Within-group variance
- Between-group variance
- Eta-squared effect size
- Post-hoc test results (Tukey's HSD) for pairwise comparisons

## Usage Examples

### Calculating Descriptive Statistics

```python
from components.statistical_analysis import perform_descriptive_stats
import pandas as pd

# Example gene expression data
data = pd.DataFrame({
    'Gene_A': [5.2, 6.1, 4.8, 5.5, 6.2, 4.9],
    'Gene_B': [2.1, 2.3, 1.8, 2.0, 2.5, 1.9],
    'Gene_C': [8.4, 9.1, 7.8, 8.2, 9.3, 8.0]
})

# Calculate descriptive statistics
stats = perform_descriptive_stats(data, ['Gene_A', 'Gene_B', 'Gene_C'])

# Display statistics
print(stats)
```

### Performing Correlation Analysis

```python
from components.statistical_analysis import perform_correlation_analysis
import pandas as pd

# Example protein-protein interaction data
data = pd.DataFrame({
    'Protein1': [0.5, 0.7, 0.3, 0.8, 0.6, 0.4],
    'Protein2': [0.6, 0.8, 0.4, 0.9, 0.7, 0.5],
    'Protein3': [0.2, 0.3, 0.1, 0.3, 0.2, 0.1]
})

# Calculate correlations and p-values
corr_matrix, p_values = perform_correlation_analysis(
    data=data,
    columns=['Protein1', 'Protein2', 'Protein3'],
    method='spearman'
)

# Display results
print("Correlation Matrix:")
print(corr_matrix)
print("\nP-values:")
print(p_values)
```

### Performing T-Test

```python
from components.statistical_analysis import perform_ttest
import pandas as pd

# Example drug response data
data = pd.DataFrame({
    'Drug_Response': [0.5, 0.7, 0.6, 0.8, 0.4, 0.9, 0.3, 0.6, 0.5, 0.7],
    'Treatment_Group': ['Control', 'Treatment', 'Treatment', 'Treatment', 'Control', 
                         'Treatment', 'Control', 'Control', 'Treatment', 'Control']
})

# Perform t-test
results = perform_ttest(
    data=data,
    numeric_column='Drug_Response',
    group_column='Treatment_Group',
    group1='Control',
    group2='Treatment',
    equal_var=True
)

# Display results
print(f"t-statistic: {results['t_statistic']:.3f}")
print(f"p-value: {results['p_value']:.4f}")
print(f"Control mean: {results['mean1']:.3f}")
print(f"Treatment mean: {results['mean2']:.3f}")
print(f"Effect size (Cohen's d): {results['effect_size']:.3f}")
print(f"Significant difference: {results['p_value'] < 0.05}")
```

## Statistical Method Selection Guide

### When to Use Each Correlation Method

| Method | Best For | Considerations |
|--------|----------|----------------|
| Pearson | Linear relationships, normally distributed data | Sensitive to outliers, assumes linearity |
| Spearman | Non-linear monotonic relationships, ordinal data | More robust to outliers, only captures monotonic relationships |
| Kendall | Small sample sizes, ordinal data | More robust than Spearman, slower to compute |

### When to Use Each Hypothesis Test

| Test | Best For | Considerations |
|------|----------|----------------|
| T-test (equal variance) | Comparing two groups with similar variance | Assumes normality, equal variance |
| T-test (unequal variance) | Comparing two groups with different variance | Assumes normality, Welch correction applied |
| ANOVA | Comparing three or more groups | Assumes normality, equal variance, followed by post-hoc tests |

## Interpreting Statistical Results

### P-values

- p < 0.05: Traditionally considered statistically significant
- p < 0.01: Strong evidence against null hypothesis
- p < 0.001: Very strong evidence against null hypothesis

However, context matters:
- Consider effect size, not just significance
- Adjust for multiple comparisons when necessary
- Biological significance may differ from statistical significance

### Effect Sizes

**Cohen's d (T-test)**:
- 0.2: Small effect
- 0.5: Medium effect
- 0.8: Large effect

**Eta-squared (ANOVA)**:
- 0.01: Small effect
- 0.06: Medium effect
- 0.14: Large effect

**Correlation Coefficients**:
- 0.1-0.3: Weak correlation
- 0.3-0.5: Moderate correlation
- 0.5-1.0: Strong correlation

## Best Practices

1. **Data Exploration**:
   - Always visualize your data before statistical testing
   - Check for normality when using parametric tests
   - Identify and handle outliers appropriately

2. **Test Selection**:
   - Choose tests based on data type and distribution
   - Consider non-parametric alternatives when assumptions are violated
   - Use appropriate corrections for multiple comparisons

3. **Interpretation**:
   - Consider effect size alongside p-values
   - Report confidence intervals when possible
   - Interpret results in the context of biological relevance

4. **Reporting**:
   - Include all relevant statistics (not just p-values)
   - Report exact p-values rather than just significance thresholds
   - Include sample sizes and power calculations where appropriate

## Common Pitfalls to Avoid

1. **P-value Hunting**: Performing multiple tests until finding significance
2. **Ignoring Assumptions**: Using parametric tests when assumptions are violated
3. **Over-reliance on P-values**: Focusing solely on significance without considering effect size
4. **Inappropriate Correlation Methods**: Using Pearson for non-linear relationships
5. **Ignoring Multiple Comparisons**: Failing to adjust p-values in multiple hypothesis testing

## Dependencies

- scipy: For statistical tests and calculations
- numpy: For numerical operations
- pandas: For data handling and statistical summaries
- statsmodels: For extended statistical functionalities

## Extension Points

The Statistical Analysis module can be extended by:
1. Adding additional statistical tests (e.g., non-parametric tests, multivariate analyses)
2. Implementing multiple comparison corrections (e.g., Bonferroni, FDR)
3. Adding power analysis and sample size calculations
4. Integrating more advanced statistical visualizations