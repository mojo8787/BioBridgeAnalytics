# Genomic API Integration

The Genomic API component provides integration with major bioinformatics databases and services for accessing and analyzing biological sequence data. This component acts as a bridge between external bioinformatics resources and the BioData Explorer application.

## Supported Databases

### NCBI Databases
- Nucleotide
- Protein
- Gene
- PubMed
- Structure

### UniProt Database
- Protein sequences
- Protein function
- Classification
- Cross-references

## Core Functions

### NCBI Integration

#### search_ncbi_databases(term, database="nucleotide", max_results=10)
Searches NCBI databases using the Entrez API.

```python
def search_ncbi_databases(term, database="nucleotide", max_results=10):
    """
    Search NCBI databases using Entrez API
    
    Args:
        term: Search term
        database: NCBI database to search (nucleotide, protein, gene, etc.)
        max_results: Maximum number of results to retrieve
        
    Returns:
        list: Search results
    """
```

#### get_sequence_record(sequence_id, database="nucleotide")
Retrieves detailed information about a specific sequence from NCBI.

```python
def get_sequence_record(sequence_id, database="nucleotide"):
    """
    Get detailed information about a specific sequence from NCBI
    
    Args:
        sequence_id: NCBI ID for the sequence
        database: NCBI database (nucleotide, protein, etc.)
        
    Returns:
        dict: Sequence record information
    """
```

### UniProt Integration

#### fetch_protein_info_from_uniprot(uniprot_id)
Fetches protein information from the UniProt API.

```python
def fetch_protein_info_from_uniprot(uniprot_id):
    """
    Fetch protein information from UniProt API
    
    Args:
        uniprot_id: UniProt accession ID
        
    Returns:
        dict: Protein information
    """
```

#### search_uniprot(query, limit=10)
Searches the UniProt database for proteins.

```python
def search_uniprot(query, limit=10):
    """
    Search UniProt database for proteins
    
    Args:
        query: Search term
        limit: Maximum number of results
        
    Returns:
        list: Search results
    """
```

### Sequence Analysis

#### analyze_sequence(sequence, analysis_type="basic")
Performs basic sequence analysis on DNA or protein sequences.

```python
def analyze_sequence(sequence, analysis_type="basic"):
    """
    Perform basic sequence analysis
    
    Args:
        sequence: DNA or protein sequence string
        analysis_type: Type of analysis to perform
        
    Returns:
        dict: Analysis results
    """
```

## API Authentication

### NCBI Entrez API
- Requires user email for API access
- Optional API key for higher request limits
- Authentication managed via `Bio.Entrez` module

### UniProt API
- No authentication required for basic access
- Rate-limited for unauthenticated users

## Usage Examples

### Searching NCBI for a Gene

```python
from components.genomic_api import search_ncbi_databases

# Search for a gene in NCBI
search_results = search_ncbi_databases(
    term="BRCA1[Gene Name] AND Homo sapiens[Organism]",
    database="gene",
    max_results=5
)

# Process search results
for result in search_results:
    print(f"Gene ID: {result['Id']}")
    print(f"Title: {result['Title']}")
```

### Retrieving Protein Information from UniProt

```python
from components.genomic_api import fetch_protein_info_from_uniprot

# Fetch protein information
protein_data = fetch_protein_info_from_uniprot("P68871")  # Hemoglobin subunit beta

# Access protein details
print(f"Protein Name: {protein_data['protein']['recommendedName']['fullName']['value']}")
print(f"Function: {protein_data['comments'][0]['text'][0]['value']}")
print(f"Length: {protein_data['sequence']['length']}")
```

### Analyzing a DNA Sequence

```python
from components.genomic_api import analyze_sequence

# Define a DNA sequence
dna_sequence = "ATGGCGACCCTGGAAAAGCTGATGAAGGCCTTCGAGTCCCTCAAGTCCTTC"

# Perform analysis
analysis_results = analyze_sequence(dna_sequence, analysis_type="basic")

# View results
print(f"Sequence Length: {analysis_results['length']}")
print(f"GC Content: {analysis_results['gc_content']:.2f}%")
print(f"Molecular Weight: {analysis_results['molecular_weight']} Da")
```

## Error Handling

The Genomic API component includes robust error handling for:
- Network connectivity issues
- API rate limits and timeouts
- Invalid sequence identifiers
- Malformed requests
- Server errors

Errors are captured, logged, and returned as user-friendly messages to aid troubleshooting.

## Dependencies

- Biopython: For Entrez API interaction and sequence handling
- Requests: For HTTP requests to UniProt and other services
- JSON: For parsing API responses

## Rate Limiting and Caching

To comply with database usage policies and improve performance:

1. NCBI requests are spaced according to their guidelines
2. Results are cached to minimize redundant API calls
3. Bulk requests are batched appropriately

## Extension Points

The Genomic API can be extended to support additional databases by:
1. Adding new API client functions
2. Implementing database-specific parsers for results
3. Integrating authentication mechanisms as required