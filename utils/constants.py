# Application information
APP_INFO = """
BioData Explorer is a data analysis platform for biologists and researchers.

The application provides tools for:
• Data loading and preprocessing
• Statistical analysis
• Visualization
• Machine learning

Created for biological data analysis.
"""

# Example use case descriptions
EXAMPLE_DESCRIPTIONS = {
    "Gene Expression Analysis": """
    Upload gene expression data (e.g., RNA-Seq or microarray) to identify
    differentially expressed genes across conditions or time points. Visualize
    expression patterns and cluster genes with similar expression profiles.
    """,
    
    "Genomic Variant Analysis": """
    Analyze VCF files to understand the distribution of genetic variants across
    samples or populations. Identify significant associations between variants and
    phenotypes or traits.
    """,
    
    "Protein Structure/Function Relationships": """
    Explore correlations between protein sequence features and their functional
    properties. Cluster proteins based on multiple features to understand structure-function
    relationships.
    """,
    
    "Microbial Community Analysis": """
    Analyze microbial abundance data from sequencing studies to understand community
    composition and diversity across different environments or conditions.
    """,
    
    "Clinical Biomarker Analysis": """
    Identify potential biomarkers in clinical datasets by finding features that
    correlate with disease status, treatment response, or other clinical outcomes.
    """
}

# File format descriptions
FILE_FORMAT_DESCRIPTIONS = {
    ".csv": "Comma-separated values file",
    ".tsv": "Tab-separated values file",
    ".txt": "Text file (tab or comma delimited)",
    ".xlsx": "Excel file",
    ".xls": "Excel file (older format)",
    ".fasta": "FASTA sequence file",
    ".fastq": "FASTQ sequence file",
    ".fa": "FASTA sequence file (alternate extension)",
    ".fna": "FASTA nucleic acid file",
    ".ffn": "FASTA nucleotide file of coding regions",
    ".faa": "FASTA amino acid file",
    ".gff": "General feature format file",
    ".vcf": "Variant call format file",
    ".bed": "BED genome annotation file",
    ".gz": "Compressed file (supports .csv.gz, .tsv.gz, etc.)"
}

# Statistical analysis descriptions
STAT_ANALYSIS_DESCRIPTIONS = {
    "Descriptive Statistics": """
    Basic statistical summary of numeric variables including mean, median, standard
    deviation, quartiles, and measures of distribution shape (skewness, kurtosis).
    """,
    
    "Correlation Analysis": """
    Measure the strength of relationships between pairs of variables. Methods include
    Pearson (linear), Spearman (rank), and Kendall (ordinal) correlations.
    """,
    
    "T-Test": """
    Compare the means of a numeric variable between two groups to determine if they
    are statistically different from each other.
    """,
    
    "ANOVA": """
    Compare the means of a numeric variable across multiple groups to identify
    statistically significant differences.
    """
}

# Machine learning analysis descriptions
ML_ANALYSIS_DESCRIPTIONS = {
    "Clustering": """
    Group similar data points together based on their features. Methods include
    K-means (spherical clusters), Hierarchical (nested clusters), and DBSCAN
    (density-based clusters).
    """,
    
    "Simple Prediction": """
    Build a model to predict a numeric value based on other variables. Methods
    include Linear Regression, Random Forest, and Support Vector Machines.
    """
}

# Visualization descriptions
VISUALIZATION_DESCRIPTIONS = {
    "Scatter Plot": """
    Visualize the relationship between two numeric variables, with optional
    color grouping by a categorical variable.
    """,
    
    "Box Plot": """
    Compare the distribution of a numeric variable across different categories,
    showing median, quartiles, and outliers.
    """,
    
    "Histogram": """
    Display the distribution of a numeric variable, showing the frequency of
    values within specific ranges (bins).
    """,
    
    "Heatmap": """
    Visualize a correlation matrix or other tabular data with color intensity
    representing numeric values.
    """,
    
    "PCA Plot": """
    Reduce the dimensionality of complex data to visualize relationships between
    samples and the influence of original variables.
    """
}
