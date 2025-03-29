# API Integration Examples

This guide provides practical examples of using BioData Explorer's API integrations to solve common biological research tasks. Follow these examples to leverage the full power of genomic and protein databases combined with AI analysis.

## NCBI Database Searches

### Finding Gene Information

**Use Case**: Search for information about a specific gene across multiple organisms.

**Steps**:
1. Navigate to "Genomic API" → "NCBI Search"
2. Select "Gene" from the database dropdown
3. Enter a search query like `BRCA1[Gene Name] AND (Homo sapiens[Organism] OR Mus musculus[Organism])`
4. Set maximum results to 10
5. Click "Search"

**Example Results**:
- Gene ID and accession numbers
- Gene location and chromosome
- Organism information
- Associated sequences
- Publication links

**Further Analysis**:
- Click on a gene ID to view detailed information
- Use "Export Results" to save gene information
- Select a gene to retrieve associated sequences

### Finding Protein Sequences

**Use Case**: Retrieve protein sequences for comparative analysis.

**Steps**:
1. Navigate to "Genomic API" → "NCBI Search"
2. Select "Protein" from the database dropdown
3. Enter a search query like `Hemoglobin[Protein Name] AND mammalia[Organism]`
4. Click "Search"
5. Select sequences of interest
6. Click "Retrieve Selected Sequences"

**Example Sequence Analysis**:
1. With the sequences retrieved, click "Analyze Sequences"
2. Select "Basic Properties" for simple analysis or "Multiple Sequence Alignment" for comparison
3. View the results showing:
   - Sequence length
   - Amino acid composition
   - Molecular weight
   - Isoelectric point
   - Alignment conservation

### Retrieving Literature on Drug Effects

**Use Case**: Find recent research on a drug's effects.

**Steps**:
1. Navigate to "Genomic API" → "NCBI Search"
2. Select "PubMed" from the database dropdown
3. Enter a search query like `"remdesivir"[Title/Abstract] AND "COVID-19"[Title/Abstract] AND ("2020"[PDAT] : "2024"[PDAT])`
4. Sort by "Most Recent"
5. Click "Search"

**Example Results Analysis**:
1. From the results, select relevant papers
2. Click "Retrieve Abstracts"
3. Use the "AI Analysis" feature to:
   - Summarize key findings
   - Extract biological entities mentioned
   - Identify common trends
4. Export a bibliography in your preferred format

## UniProt Protein Analysis

### Detailed Protein Information

**Use Case**: Obtain comprehensive information about a specific protein.

**Steps**:
1. Navigate to "Genomic API" → "UniProt Search"
2. Enter a UniProt accession ID (e.g., P68871 for human hemoglobin subunit beta)
3. Click "Retrieve Information"

**Example Information Displayed**:
- Protein name and aliases
- Sequence information
- Function description
- Subcellular location
- Tissue specificity
- Post-translational modifications
- Disease associations
- 3D structure information

**Further Analysis Options**:
1. Click "View Sequence" to examine the amino acid sequence
2. Select "Analyze with AI" to predict additional properties
3. Choose "View 3D Structure" to open a 3D visualization (if available)

### Searching Proteins by Function

**Use Case**: Find proteins involved in a specific biological process.

**Steps**:
1. Navigate to "Genomic API" → "UniProt Search"
2. Enter a functional query like `antioxidant activity AND reviewed:yes`
3. Set organism to "Homo sapiens"
4. Click "Search"

**Example Follow-up Analysis**:
1. From the results, select proteins of interest
2. Click "Compare Selected Proteins" to see:
   - Sequence similarity
   - Domain architecture comparison
   - Functional annotation comparison
   - Phylogenetic relationships

## AI-Powered Sequence Analysis

### Protein Function Prediction

**Use Case**: Predict the function of an uncharacterized protein sequence.

**Steps**:
1. Navigate to "AI Analysis" → "Sequence Analysis"
2. Paste a protein sequence or upload a FASTA file
3. Select "Function Prediction" as analysis type
4. Choose model (e.g., "ESM-2" or "OpenAI")
5. Click "Analyze"

**Example Output**:
- Predicted molecular function
- Possible biological processes
- Potential cellular components
- Confidence scores for each prediction
- Similar proteins with known functions

**How to Interpret**:
- Focus on predictions with high confidence scores
- Cross-reference with BLAST search results
- Consider evolutionary conservation information
- Use predicted domains to narrow down function

### Predicting Protein-Protein Interactions

**Use Case**: Predict potential interaction partners for a protein.

**Steps**:
1. Navigate to "AI Analysis" → "Protein Interaction"
2. Enter a protein identifier or paste a sequence
3. Select search scope (e.g., "Human Proteome")
4. Click "Predict Interactions"

**Example Results**:
- Ranked list of potential interaction partners
- Interaction confidence scores
- Predicted binding sites
- Biological context for interactions
- Supporting evidence from literature

## Research Paper Analysis

### Summarizing Scientific Papers

**Use Case**: Extract key information from a long research paper.

**Steps**:
1. Navigate to "AI Analysis" → "Research Paper Analysis"
2. Paste the paper text or upload a PDF/text file
3. Select "Comprehensive Summary" as analysis type
4. Click "Analyze"

**Example Output**:
- Executive summary (1-2 paragraphs)
- Key findings highlights
- Methodology summary
- Results interpretation
- Limitations mentioned
- Future research directions

### Extracting Biological Entities

**Use Case**: Identify all biological entities mentioned in research text.

**Steps**:
1. Navigate to "AI Analysis" → "Research Paper Analysis"
2. Paste text or upload a file
3. Select "Entity Extraction" as analysis type
4. Click "Analyze"

**Example Extracted Categories**:
- Genes and gene products
- Proteins and protein families
- Chemical compounds and drugs
- Diseases and phenotypes
- Cellular components
- Biological processes
- Experimental methods

**Further Analysis**:
1. Click "Create Entity Network" to visualize relationships
2. Select "Find Related Literature" to discover connected research
3. Choose "Export Entities" to save the structured data

## Combining Multiple APIs

### Integrated Analysis Workflow

**Use Case**: Comprehensive analysis of a disease-associated gene.

**Workflow**:

1. **Find Disease Gene**:
   - Use "NCBI Search" to find genes associated with a disease
   - Select a gene of interest (e.g., CFTR for cystic fibrosis)

2. **Retrieve Protein Information**:
   - Get the protein sequence and information from UniProt
   - Note functional domains and known variants

3. **Analyze Variants**:
   - Use "Sequence Analysis" to examine how variants affect protein function
   - Predict functional impact of mutations with AI tools

4. **Literature Synthesis**:
   - Search PubMed for recent research on this gene/protein
   - Use AI to summarize findings and identify trends

5. **Generate Report**:
   - Compile all findings into a comprehensive report
   - Include visualizations and predictions
   - Export as a shareable HTML document

## API Usage Tips

### Optimizing Searches

For more effective database searches:
- Use Boolean operators (AND, OR, NOT) to refine results
- Put phrases in quotes for exact matching
- Use field tags to search specific attributes ([Gene Name], [Organism], etc.)
- Include date restrictions to focus on recent research
- Use wildcard characters (*) for partial matching

### Managing Large Results

When dealing with large result sets:
- Use more specific search terms to narrow results
- Add filters for organism, review status, or feature type
- Increase specificity by combining search terms
- Use sorting options to prioritize most relevant results
- Download results in batches if necessary

### Saving API Configurations

To save time on repeated analyses:
- Use the "Save Configuration" option to store API settings
- Create named presets for common search patterns
- Export search strategies for sharing with colleagues
- Note successful query structures for future reference

## Troubleshooting

### Common API Issues

| Issue | Solution |
|-------|----------|
| No search results | Try broader terms, check spelling, remove filters |
| Too many results | Add more specific terms, use additional filters |
| Slow API response | Reduce result size, try at different time of day |
| Blocked IP | Ensure you're not exceeding usage limits |
| Parsing errors | Check that sequence format is correct |

### Getting Support

If you encounter persistent issues:
1. Check the API status indicator in the application
2. Review the specific error message for guidance
3. Consult the official database documentation
4. Contact the developer for specialized assistance

---

Remember, the real power of BioData Explorer comes from combining these different API capabilities into integrated analysis workflows tailored to your research questions.