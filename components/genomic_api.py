import streamlit as st
import requests
import json
import tempfile
import os
import pandas as pd
from Bio import Entrez, SeqIO
from io import StringIO

# Set your email for NCBI Entrez
Entrez.email = "your_email@example.com"  # This will be overridden by user input

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
    try:
        # Perform the search
        search_handle = Entrez.esearch(db=database, term=term, retmax=max_results)
        search_results = Entrez.read(search_handle)
        search_handle.close()
        
        # Get the list of IDs
        id_list = search_results["IdList"]
        
        if not id_list:
            return []
        
        # Fetch records for these IDs
        fetch_handle = Entrez.efetch(db=database, id=id_list, rettype="gb", retmode="text")
        records = fetch_handle.read()
        fetch_handle.close()
        
        # Return the records
        return {"records": records, "id_list": id_list}
    
    except Exception as e:
        st.error(f"Error searching NCBI: {str(e)}")
        return []

def get_sequence_record(sequence_id, database="nucleotide"):
    """
    Get detailed information about a specific sequence from NCBI
    
    Args:
        sequence_id: NCBI ID for the sequence
        database: NCBI database (nucleotide, protein, etc.)
        
    Returns:
        dict: Sequence record information
    """
    try:
        # Fetch the record
        handle = Entrez.efetch(db=database, id=sequence_id, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()
        
        # Extract information
        info = {
            "id": record.id,
            "name": record.name,
            "description": record.description,
            "sequence": str(record.seq),
            "length": len(record.seq),
            "annotations": dict(record.annotations),
            "features": [
                {
                    "type": feature.type,
                    "location": str(feature.location),
                    "qualifiers": dict(feature.qualifiers)
                }
                for feature in record.features
            ]
        }
        
        return info
    
    except Exception as e:
        st.error(f"Error fetching sequence record: {str(e)}")
        return None

def fetch_protein_info_from_uniprot(uniprot_id):
    """
    Fetch protein information from UniProt API
    
    Args:
        uniprot_id: UniProt accession ID
        
    Returns:
        dict: Protein information
    """
    try:
        # Construct API URL
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        
        # Make the request
        response = requests.get(url)
        
        # Check if request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            # Extract relevant information
            protein_info = {
                "id": data.get("primaryAccession", ""),
                "name": data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                "gene": data.get("genes", [{}])[0].get("geneName", {}).get("value", ""),
                "organism": data.get("organism", {}).get("scientificName", ""),
                "sequence": data.get("sequence", {}).get("value", ""),
                "length": data.get("sequence", {}).get("length", 0),
                "function": next((comment.get("texts", [{}])[0].get("value", "") 
                            for comment in data.get("comments", []) 
                            if comment.get("commentType") == "FUNCTION"), ""),
                "features": [
                    {
                        "type": feature.get("type", ""),
                        "location": feature.get("location", {}),
                        "description": feature.get("description", "")
                    }
                    for feature in data.get("features", [])
                ]
            }
            
            return protein_info
        else:
            st.error(f"Error fetching protein info: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Error connecting to UniProt API: {str(e)}")
        return None

def search_uniprot(query, limit=10):
    """
    Search UniProt database for proteins
    
    Args:
        query: Search term
        limit: Maximum number of results
        
    Returns:
        list: Search results
    """
    try:
        # Construct API URL
        url = "https://rest.uniprot.org/uniprotkb/search"
        
        # Parameters for the search
        params = {
            "query": query,
            "format": "json",
            "size": limit
        }
        
        # Make the request
        response = requests.get(url, params=params)
        
        # Check if request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            # Extract relevant information from each result
            results = [
                {
                    "id": item.get("primaryAccession", ""),
                    "entry_name": item.get("uniProtkbId", ""),
                    "protein_name": item.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                    "gene": item.get("genes", [{}])[0].get("geneName", {}).get("value", "") if item.get("genes") else "",
                    "organism": item.get("organism", {}).get("scientificName", ""),
                    "length": item.get("sequence", {}).get("length", 0)
                }
                for item in data.get("results", [])
            ]
            
            return results
        else:
            st.error(f"Error searching UniProt: {response.status_code}")
            return []
            
    except Exception as e:
        st.error(f"Error connecting to UniProt API: {str(e)}")
        return []

def analyze_sequence(sequence, analysis_type="basic"):
    """
    Perform basic sequence analysis
    
    Args:
        sequence: DNA or protein sequence string
        analysis_type: Type of analysis to perform
        
    Returns:
        dict: Analysis results
    """
    from Bio.SeqUtils import GC, molecular_weight
    from Bio.Seq import Seq
    
    try:
        seq = Seq(sequence)
        
        # Basic analysis
        results = {
            "length": len(seq),
            "gc_content": GC(seq),
            "molecular_weight": molecular_weight(seq),
        }
        
        # For DNA sequences, add transcription and translation
        if all(c in 'ATGCatgc' for c in sequence):
            results["is_dna"] = True
            results["transcription"] = str(seq.transcribe())
            results["translation"] = str(seq.translate())
            
            # Count nucleotides
            nucleotide_counts = {
                'A': sequence.upper().count('A'),
                'T': sequence.upper().count('T'),
                'G': sequence.upper().count('G'),
                'C': sequence.upper().count('C')
            }
            results["nucleotide_counts"] = nucleotide_counts
        else:
            results["is_dna"] = False
            
            # Count amino acids
            from Bio.SeqUtils.ProtParam import ProteinAnalysis
            try:
                # Remove invalid characters
                clean_seq = ''.join(c for c in sequence if c in "ACDEFGHIKLMNPQRSTVWY")
                prot_analysis = ProteinAnalysis(clean_seq)
                results["amino_acid_counts"] = prot_analysis.count_amino_acids()
                results["isoelectric_point"] = prot_analysis.isoelectric_point()
                results["gravy"] = prot_analysis.gravy()  # Grand average of hydropathy
                results["aromaticity"] = prot_analysis.aromaticity()
                results["instability_index"] = prot_analysis.instability_index()
                results["secondary_structure_fraction"] = prot_analysis.secondary_structure_fraction()
            except Exception as e:
                st.warning(f"Some protein analyses could not be completed: {str(e)}")
        
        return results
        
    except Exception as e:
        st.error(f"Error analyzing sequence: {str(e)}")
        return None