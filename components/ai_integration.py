import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import openai
from io import StringIO

def analyze_text_with_openai(text, prompt, max_tokens=1000):
    """
    Use OpenAI to analyze text data
    
    Args:
        text: Text to analyze
        prompt: Specific prompt for OpenAI
        max_tokens: Maximum number of tokens in the response
        
    Returns:
        str: OpenAI response
    """
    # Ensure API key is set
    if not openai.api_key:
        st.error("OpenAI API key not set")
        return None
    
    try:
        # Create the full prompt
        full_prompt = f"{prompt}\n\nText to analyze:\n{text}"
        
        # Call the OpenAI API
        response = openai.completions.create(
            model="gpt-3.5-turbo-instruct",  # or gpt-4 if available
            prompt=full_prompt,
            max_tokens=max_tokens,
            temperature=0.2  # Lower temperature for more focused responses
        )
        
        # Return the response text
        return response.choices[0].text
    
    except Exception as e:
        st.error(f"Error using OpenAI API: {str(e)}")
        return None

def summarize_research_paper(text, focus_area=None):
    """
    Summarize a research paper using OpenAI
    
    Args:
        text: Research paper text
        focus_area: Optional specific area to focus on
        
    Returns:
        str: Summary of the paper
    """
    # Create a prompt focused on scientific research
    if focus_area:
        prompt = f"""Summarize the following scientific research paper with a focus on {focus_area}. 
        Include: 
        - Main research question
        - Methodology
        - Key findings
        - Implications for the field of {focus_area}
        - Limitations of the study
        
        Format the summary in clear sections with headings."""
    else:
        prompt = """Summarize the following scientific research paper.
        Include:
        - Main research question and objectives
        - Methodology and approach
        - Key findings and results
        - Conclusions and implications
        - Limitations of the study
        
        Format the summary in clear sections with headings."""
    
    return analyze_text_with_openai(text, prompt, max_tokens=1500)

def extract_biological_entities(text):
    """
    Extract biological entities from text using OpenAI
    
    Args:
        text: Text to analyze
        
    Returns:
        dict: Extracted entities by category
    """
    prompt = """Extract all biological entities from the text and categorize them. 
    Include the following categories:
    - Genes and proteins
    - Organisms and species
    - Diseases and conditions
    - Chemical compounds and drugs
    - Biological processes
    - Laboratory techniques
    
    Format the output as a structured JSON with these categories as keys and lists of found entities as values.
    Only return the JSON, no other text."""
    
    result = analyze_text_with_openai(text, prompt)
    
    if result:
        try:
            # Clean up the response to extract valid JSON
            json_str = result.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].strip()
            
            # Parse the JSON
            entities = json.loads(json_str)
            return entities
        except Exception as e:
            st.error(f"Error parsing entity extraction results: {str(e)}")
            return {"error": str(e), "raw_result": result}
    
    return None

def analyze_sequence_with_ai(sequence, analysis_type="function_prediction"):
    """
    Analyze biological sequence using AI
    
    Args:
        sequence: Biological sequence (DNA, RNA, or protein)
        analysis_type: Type of analysis to perform
        
    Returns:
        dict: Analysis results
    """
    # Determine sequence type
    seq_type = "DNA/RNA" if set(sequence.upper()) <= set("ATGCUN") else "protein"
    
    if analysis_type == "function_prediction" and seq_type == "protein":
        prompt = """Analyze this protein sequence and predict its possible functions.
        Consider:
        - Any recognizable domains or motifs
        - Potential cellular localization
        - Possible biochemical activities
        - Structural characteristics
        
        Format the output as a structured JSON with these categories:
        - predicted_functions: list of possible functions
        - confidence: confidence level for each prediction (high/medium/low)
        - rationale: brief explanation for each prediction
        
        Only return the JSON, no other text."""
    
    elif analysis_type == "structural_prediction" and seq_type == "protein":
        prompt = """Provide insights about the potential structural characteristics of this protein sequence.
        Consider:
        - Secondary structure elements (alpha helices, beta sheets)
        - Potential domains
        - Structural motifs
        - Possible 3D conformations
        
        Format the output as a structured JSON with these categories:
        - secondary_structure: prediction of alpha/beta content
        - domains: possible domains
        - motifs: structural motifs identified
        - stability: prediction about the protein stability
        
        Only return the JSON, no other text."""
    
    elif seq_type == "DNA/RNA":
        prompt = """Analyze this DNA/RNA sequence and provide insights.
        Consider:
        - Potential genes or open reading frames
        - Regulatory elements
        - Sequence characteristics
        - Possible functions
        
        Format the output as a structured JSON with these categories:
        - sequence_type: DNA or RNA
        - potential_genes: possible genes or ORFs
        - regulatory_elements: possible regulatory elements
        - characteristics: notable sequence characteristics
        
        Only return the JSON, no other text."""
    
    else:
        prompt = """Analyze this biological sequence and provide detailed insights about its characteristics and potential functions.
        Format the output as a structured JSON with relevant categories.
        Only return the JSON, no other text."""
    
    result = analyze_text_with_openai(sequence, prompt)
    
    if result:
        try:
            # Clean up the response to extract valid JSON
            json_str = result.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].strip()
            
            # Parse the JSON
            analysis = json.loads(json_str)
            return analysis
        except Exception as e:
            st.error(f"Error parsing sequence analysis results: {str(e)}")
            return {"error": str(e), "raw_result": result}
    
    return None

def huggingface_protein_prediction(sequence, model_id="facebook/esm2_t33_650M_UR50D"):
    """
    Mock implementation of protein prediction that would normally use Hugging Face
    
    Args:
        sequence: Protein sequence
        model_id: Hugging Face model ID
        
    Returns:
        dict: Prediction results
    """
    try:
        # Basic validation for protein sequence
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(aa in valid_aa for aa in sequence.upper()):
            st.warning("The sequence contains characters not found in standard protein sequences.")
        
        # For demonstration, we'll just calculate some basic stats about the sequence
        # In a real implementation, this would use the Hugging Face API
        st.info("Note: This is a simplified implementation. In production, this would use the Hugging Face API.")
        
        # Calculate amino acid frequencies
        aa_freqs = {}
        for aa in valid_aa:
            aa_freqs[aa] = sequence.upper().count(aa) / len(sequence) if sequence else 0
        
        # Calculate simple embedding statistics (simulated)
        result = {
            "sequence_length": len(sequence),
            "unique_amino_acids": len(set(sequence.upper())),
            "most_common_aa": max(aa_freqs.items(), key=lambda x: x[1])[0] if aa_freqs else None,
            "model_id": model_id,
            "note": "This is a simulated result without actual model inference"
        }
        
        return result
    
    except Exception as e:
        st.error(f"Error in protein prediction: {str(e)}")
        return None