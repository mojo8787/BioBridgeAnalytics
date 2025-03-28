import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from Bio import SeqIO
import tempfile
import gzip

def get_supported_formats():
    """
    Returns a dictionary of supported file formats and their descriptions.
    
    Returns:
        dict: Dictionary of file extensions and descriptions
    """
    return {
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

def load_data(uploaded_file):
    """
    Load data from an uploaded file into a pandas DataFrame or appropriate format.
    
    Args:
        uploaded_file: Streamlit's UploadedFile object
        
    Returns:
        DataFrame or dict: Loaded data in appropriate format
    """
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    # Handle compressed files
    if file_ext == ".gz":
        # Get the actual extension before .gz
        base_name = os.path.splitext(os.path.splitext(uploaded_file.name)[0])[1].lower()
        if base_name in [".csv", ".tsv", ".txt"]:
            # Create a temporary file to write the decompressed content
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(gzip.decompress(uploaded_file.getvalue()))
            
            # Read the decompressed file
            if base_name == ".csv":
                data = pd.read_csv(tmp.name)
            elif base_name in [".tsv", ".txt"]:
                data = pd.read_csv(tmp.name, sep="\t")
            
            # Clean up the temporary file
            os.unlink(tmp.name)
            return data
    
    # Handle standard file formats
    if file_ext in [".csv"]:
        return pd.read_csv(uploaded_file)
    
    elif file_ext in [".tsv", ".txt"]:
        return pd.read_csv(uploaded_file, sep="\t")
    
    elif file_ext in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)
    
    elif file_ext in [".fasta", ".fa", ".fna", ".ffn", ".faa"]:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Parse FASTA file using Biopython
        sequences = list(SeqIO.parse(tmp_path, "fasta"))
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        # Convert to DataFrame with sequence info
        data = pd.DataFrame({
            'id': [seq.id for seq in sequences],
            'description': [seq.description for seq in sequences],
            'sequence': [str(seq.seq) for seq in sequences],
            'length': [len(seq.seq) for seq in sequences]
        })
        
        return data
    
    elif file_ext == ".fastq":
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Parse FASTQ file using Biopython
        sequences = list(SeqIO.parse(tmp_path, "fastq"))
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        # Convert to DataFrame with sequence info
        data = pd.DataFrame({
            'id': [seq.id for seq in sequences],
            'description': [seq.description for seq in sequences],
            'sequence': [str(seq.seq) for seq in sequences],
            'quality': [seq.letter_annotations['phred_quality'] for seq in sequences],
            'length': [len(seq.seq) for seq in sequences]
        })
        
        return data
    
    elif file_ext == ".gff":
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
            
        # Read GFF file, skipping comment lines
        data = pd.read_csv(
            tmp_path,
            sep='\t',
            comment='#',
            names=['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']
        )
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        return data
    
    elif file_ext == ".vcf":
        # Read VCF file with pandas
        # Create a temporary file to write content
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
            
        # Read the file skipping the header lines that start with ##
        with open(tmp_path, 'r') as file:
            vcf_lines = [line for line in file if not line.startswith('##')]
        
        # Parse the VCF content
        if vcf_lines:
            # If there's a header line starting with single #, use it as the header
            if vcf_lines[0].startswith('#'):
                header = vcf_lines[0].strip('#').strip().split('\t')
                data = pd.read_csv(io.StringIO(''.join(vcf_lines[1:])), sep='\t', names=header)
            else:
                # Default VCF columns if no header
                default_cols = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT']
                data = pd.read_csv(io.StringIO(''.join(vcf_lines)), sep='\t', names=default_cols)
        else:
            # Empty dataframe with VCF columns if file is empty or only has ## lines
            data = pd.DataFrame(columns=['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO'])
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        return data
    
    elif file_ext == ".bed":
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
            
        # Read BED file
        try:
            # Try with the standard 3-column BED format
            data = pd.read_csv(
                tmp_path,
                sep='\t',
                comment='#',
                header=None,
                names=['chrom', 'chromStart', 'chromEnd']
            )
        except pd.errors.ParserError:
            # Try with extended BED format (more columns)
            try:
                data = pd.read_csv(
                    tmp_path,
                    sep='\t',
                    comment='#',
                    header=None
                )
                
                # Name columns according to BED format
                if data.shape[1] >= 3:
                    col_names = ['chrom', 'chromStart', 'chromEnd']
                    if data.shape[1] >= 4:
                        col_names.append('name')
                    if data.shape[1] >= 5:
                        col_names.append('score')
                    if data.shape[1] >= 6:
                        col_names.append('strand')
                    if data.shape[1] >= 7:
                        col_names.append('thickStart')
                    if data.shape[1] >= 8:
                        col_names.append('thickEnd')
                    if data.shape[1] >= 9:
                        col_names.append('itemRgb')
                    if data.shape[1] >= 10:
                        col_names.append('blockCount')
                    if data.shape[1] >= 11:
                        col_names.append('blockSizes')
                    if data.shape[1] >= 12:
                        col_names.append('blockStarts')
                    
                    # Add generic column names for any additional columns
                    for i in range(len(col_names), data.shape[1]):
                        col_names.append(f'column{i+1}')
                    
                    data.columns = col_names
            
            except Exception as e:
                st.error(f"Error parsing BED file: {str(e)}")
                return None
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        return data
    
    else:
        st.error(f"Unsupported file format: {file_ext}")
        return None
