# Data Loader Component

The Data Loader component provides functionality for loading various biological data formats into the BioData Explorer application. It includes automatic format detection, data validation, and specialized handling for different biological data types.

## Supported Data Formats

| Format | Extensions | Description |
|--------|------------|-------------|
| CSV | .csv | Comma-separated values |
| Excel | .xlsx, .xls | Microsoft Excel workbooks |
| TSV | .tsv, .txt | Tab-separated values |
| FASTA | .fasta, .fa, .fna, .ffn, .faa, .frn | Sequence data format |
| VCF | .vcf | Variant Call Format for genetic variations |
| GenBank | .gb, .gbk | GenBank sequence format |
| Plain Text | .txt | Plain text files |

## Core Functions

### get_supported_formats()
Returns a dictionary of supported file formats and their descriptions.

```python
def get_supported_formats():
    """
    Returns a dictionary of supported file formats and their descriptions.
    
    Returns:
        dict: Dictionary of file extensions and descriptions
    """
    return {
        ".csv": "Comma-separated values",
        ".xlsx": "Excel workbook",
        ".xls": "Excel workbook (legacy)",
        ".tsv": "Tab-separated values",
        ".txt": "Text file (tab-delimited or plain text)",
        ".fasta": "FASTA sequence format",
        ".fa": "FASTA sequence format",
        ".fna": "FASTA nucleic acid",
        ".ffn": "FASTA nucleotide coding regions",
        ".faa": "FASTA amino acid",
        ".frn": "FASTA non-coding RNA",
        ".vcf": "Variant Call Format",
        ".gb": "GenBank format",
        ".gbk": "GenBank format"
    }
```

### load_data(uploaded_file)
Loads data from an uploaded file into a pandas DataFrame or other appropriate format.

```python
def load_data(uploaded_file):
    """
    Load data from an uploaded file into a pandas DataFrame or appropriate format.
    
    Args:
        uploaded_file: Streamlit's UploadedFile object
        
    Returns:
        DataFrame or dict: Loaded data in appropriate format
    """
```

#### Handling Different File Types

**Tabular Data**:
- CSV, TSV: Loaded as pandas DataFrames
- Excel: Loaded as pandas DataFrames with sheet selection options

**Sequence Data**:
- FASTA: Processed with BioPython's SeqIO module
- GenBank: Processed with BioPython's SeqIO module

**Variant Data**:
- VCF: Converted to a pandas DataFrame with specialized parsing

**Text Data**:
- TXT: Loaded as plain text or attempted as TSV if structured

## Usage Examples

### Loading a CSV File

```python
from components.data_loader import load_data

# Assuming uploaded_file is a Streamlit UploadedFile
data = load_data(uploaded_file)

# Access data as a pandas DataFrame
print(data.head())
print(data.columns)
```

### Working with FASTA Data

```python
from components.data_loader import load_data

# Assuming uploaded_file is a FASTA file
sequence_data = load_data(uploaded_file)

# Access the sequence records
for record_id, record in sequence_data.items():
    print(f"Sequence ID: {record_id}")
    print(f"Sequence: {record.seq}")
    print(f"Description: {record.description}")
```

## Error Handling

The Data Loader provides robust error handling for:
- Invalid or corrupted files
- Unsupported formats
- Encoding issues
- Parse errors in structured formats

Errors are captured and returned as user-friendly messages to guide troubleshooting.

## Dependencies

- pandas: For loading tabular data
- Bio.SeqIO: For processing sequence files
- io: For file stream handling
- streamlit: For UI integration

## Extension Points

The Data Loader can be extended to support additional formats by:
1. Adding new format entries to `get_supported_formats()`
2. Implementing format-specific loading logic in `load_data()`