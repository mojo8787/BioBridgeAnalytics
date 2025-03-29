# BioData Explorer

## Overview

BioData Explorer is a comprehensive biological data analysis platform developed by AlMotasem Bellah Younis, PhD. The application provides researchers, biologists, and data scientists with powerful tools to analyze, visualize, and extract insights from complex biological datasets without requiring extensive programming knowledge.

## Developer

**AlMotasem Bellah Younis, PhD**
- Researcher in nanoparticle & antimicrobial studies
- PhD in Biochemistry from Mendel University
- Specializes in Biotechnology, Microbiology, Data Science, and Machine Learning applications in biological research

## Features

### Core Functionality

1. **Data Loading and Processing**
   - Support for multiple biological data formats (CSV, Excel, FASTA, VCF, etc.)
   - Automatic data preprocessing and cleaning
   - Handling of missing values
   - Data filtering options

2. **Statistical Analysis**
   - Descriptive statistics
   - Correlation analysis (Pearson, Spearman, Kendall methods)
   - Hypothesis testing (T-tests, ANOVA)
   - Statistical summaries and interpretations

3. **Data Visualization**
   - Interactive scatter plots
   - Box plots for categorical comparisons
   - Histograms for distribution analysis
   - Correlation heatmaps
   - PCA (Principal Component Analysis) plots for dimensionality reduction

4. **Machine Learning Tools**
   - Clustering algorithms (K-means, Hierarchical, DBSCAN)
   - Prediction models (Linear Regression, Random Forest, SVM)
   - Feature importance analysis
   - Model performance metrics

5. **Genomic and Protein API Integration**
   - NCBI database integration for nucleotide and protein sequences
   - UniProt API for protein information
   - Sequence analysis tools
   - Database searching capabilities

6. **AI Integration**
   - Research paper analysis and summarization using OpenAI
   - Biological entity extraction from text
   - Sequence analysis with AI
   - Protein language model integration (ESM models)

7. **Export Capabilities**
   - HTML report generation
   - CSV export
   - Excel workbook creation
   - Figure saving in various formats

## Technical Architecture

### Application Structure

```
├── .streamlit/              # Streamlit configuration
│   └── config.toml          # Server and theme settings
├── components/              # Core application components
│   ├── ai_integration.py    # AI/ML integration with OpenAI and protein models
│   ├── data_loader.py       # Data loading functionality for different formats
│   ├── data_processor.py    # Data preprocessing and transformation
│   ├── file_handler.py      # File operations (saving, exporting)
│   ├── genomic_api.py       # Integration with NCBI and UniProt APIs
│   ├── ml_analysis.py       # Machine learning algorithms
│   ├── statistical_analysis.py # Statistical methods
│   └── visualization.py     # Data visualization functions
├── utils/                   # Utility modules
│   ├── constants.py         # Application constants and text
│   └── helpers.py           # Helper functions
├── app.py                   # Main application file
```

### Technologies Used

- **Streamlit**: Web application framework
- **Pandas & NumPy**: Data handling and processing
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning algorithms
- **SciPy**: Statistical analysis
- **BioPython**: Biological sequence handling
- **OpenAI API**: Text analysis and AI integration
- **Requests**: API interactions

## Modules Details

### app.py
The main application file containing the Streamlit UI components, session state management, and navigation logic. Integrates all the components to create the full application.

### data_loader.py
Handles loading data from various file formats. Supports:
- Tabular data (CSV, Excel, TSV)
- Sequence data (FASTA)
- Variant data (VCF)
- Text data

### data_processor.py
Performs data preprocessing tasks:
- Cleaning and transformation
- Column type identification
- Missing value detection
- Correlation identification

### visualization.py
Contains functions for creating various types of visualizations:
- Scatter plots with optional coloring
- Box plots for categorical comparisons
- Histograms for distributions
- Correlation heatmaps
- PCA plots with explained variance

### statistical_analysis.py
Implements statistical methods:
- Descriptive statistics (mean, median, std, etc.)
- Correlation analysis with p-values
- T-tests for group comparisons
- ANOVA for multi-group analysis

### ml_analysis.py
Provides machine learning functionality:
- Clustering methods with parameter tuning
- Prediction models for regression tasks
- Feature importance analysis
- Model performance evaluation

### genomic_api.py
Integrates with biological databases:
- NCBI/GenBank search and retrieval
- UniProt protein information access
- Sequence analysis utilities
- API request handling

### ai_integration.py
Connects to AI services:
- OpenAI for text analysis
- Research paper summarization
- Biological entity extraction
- Protein language model integration

### file_handler.py
Manages file operations:
- Saving visualizations
- Exporting analysis results
- Report generation
- Data export in various formats

### constants.py
Contains application constants:
- Application information
- Analysis descriptions
- File format specifications
- Help text and documentation

### helpers.py
Utility functions:
- Data formatting
- Summary statistics
- Text processing
- Data type inference

## UI Components

### Navigation
- Sidebar navigation for different sections
- Tab-based interface for analyses

### Data Upload Section
- File uploader with format validation
- Data preview and summary
- Column type information

### Data Analysis Tools
- Statistical analysis options
- Visualization tools with parameters
- Machine learning model configuration

### API Integration
- Configuration for genomic and protein APIs
- Search interfaces for biological databases
- API key management

### AI Tools
- Research paper analysis interface
- Entity extraction from text
- Sequence analysis with AI

### Export Options
- Format selection for exports
- Report customization
- Download buttons for results

## Usage Instructions

1. **Data Upload**
   - Select a data file in a supported format
   - Review the data summary and column information
   - Handle missing values if needed

2. **Data Analysis**
   - Choose the analysis type from the navigation menu
   - Select columns and parameters for the analysis
   - View and interpret the results

3. **Visualization**
   - Select a visualization type
   - Configure the plot parameters
   - Download or save visualizations

4. **Machine Learning**
   - Choose a clustering or prediction method
   - Configure model parameters
   - Evaluate model performance

5. **Genomic/Protein API**
   - Enter API configuration details
   - Search databases for sequences
   - Analyze retrieved sequences

6. **AI Integration**
   - Configure API keys for AI services
   - Upload or enter text for analysis
   - Review AI-generated insights

7. **Export Results**
   - Select export format
   - Configure export options
   - Download results

## System Requirements

- Python 3.8+
- Modern web browser
- Internet connection for API features

## Dependencies

- streamlit
- pandas
- numpy
- plotly
- scikit-learn
- scipy
- biopython
- requests
- openai

## Getting Started

1. Clone the repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`
4. Access the application in your browser at http://localhost:5000

## API Integration

### NCBI/GenBank
- Requires user email for API access
- Used for nucleotide and protein sequence retrieval
- Provides sequence analysis functionality

### UniProt
- Public API for protein information
- No authentication required for basic access
- Provides comprehensive protein data

### OpenAI
- Requires API key
- Used for text analysis and biological entity extraction
- Powers research paper summarization

## Future Developments

- Additional machine learning algorithms
- Enhanced visualization options
- Expanded API integrations
- Support for more file formats
- Collaborative analysis features

---

## Copyright

Copyright © 2024 AlMotasem Bellah Younis. All rights reserved.