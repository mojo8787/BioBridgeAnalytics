import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import json
import plotly.io as pio
import base64
from datetime import datetime

def save_figure(fig, filename_base, format='html'):
    """
    Save a Plotly figure to a file.
    
    Args:
        fig: Plotly figure to save
        filename_base: Base name for the output file (without extension)
        format: Output format (html, png, jpg, svg, pdf)
        
    Returns:
        str: Path to the saved file
    """
    # Create a temporary directory to save the file
    temp_dir = tempfile.mkdtemp()
    
    # Create the full path with the appropriate extension
    file_path = os.path.join(temp_dir, f"{filename_base}.{format}")
    
    # Save the figure
    if format == 'html':
        pio.write_html(fig, file=file_path)
    else:
        pio.write_image(fig, file=file_path)
    
    return file_path

def export_analysis_report(data, analysis_results, report_title="BioData Analysis Report", include_visualizations=True):
    """
    Export the analysis results as an HTML report.
    
    Args:
        data: DataFrame with the dataset
        analysis_results: Dictionary with analysis results
        report_title: Title for the report
        include_visualizations: Whether to include visualizations in the report
        
    Returns:
        str: Path to the generated HTML report
    """
    # Create a temporary file for the report
    temp_dir = tempfile.mkdtemp()
    report_path = os.path.join(temp_dir, "analysis_report.html")
    
    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start building the HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{report_title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3, h4 {{
                color: #2c3e50;
                margin-top: 20px;
            }}
            h1 {{
                text-align: center;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
                font-size: 0.8em;
                color: #7f8c8d;
            }}
            .visualization {{
                margin: 20px 0;
                text-align: center;
            }}
            .timestamp {{
                text-align: right;
                font-style: italic;
                color: #7f8c8d;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{report_title}</h1>
            <p class="timestamp">Generated on: {current_datetime}</p>
            
            <h2>Dataset Overview</h2>
            <p>Number of rows: {len(data)}</p>
            <p>Number of columns: {len(data.columns)}</p>
            
            <h3>Data Preview</h3>
            {data.head(10).to_html()}
    """
    
    # Add analysis results sections
    if "descriptive_stats" in analysis_results:
        html_content += f"""
            <h2>Descriptive Statistics</h2>
            {analysis_results["descriptive_stats"].to_html()}
        """
    
    if "correlation" in analysis_results:
        html_content += f"""
            <h2>Correlation Analysis</h2>
            <h3>Correlation Matrix ({analysis_results["correlation"]["method"]})</h3>
            {analysis_results["correlation"]["matrix"].to_html()}
            
            <h3>P-Value Matrix</h3>
            {analysis_results["correlation"]["p_values"].to_html()}
        """
    
    if "ttest" in analysis_results:
        t_results = analysis_results["ttest"]["results"]
        html_content += f"""
            <h2>T-Test Analysis</h2>
            <p>Comparing <strong>{analysis_results["ttest"]["variable"]}</strong> between groups: 
            <strong>{analysis_results["ttest"]["groups"][0]}</strong> and <strong>{analysis_results["ttest"]["groups"][1]}</strong></p>
            
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>t-statistic</td>
                    <td>{t_results["t_statistic"]:.4f}</td>
                </tr>
                <tr>
                    <td>p-value</td>
                    <td>{t_results["p_value"]:.4f}</td>
                </tr>
                <tr>
                    <td>Mean (Group 1)</td>
                    <td>{t_results["mean1"]:.4f}</td>
                </tr>
                <tr>
                    <td>Mean (Group 2)</td>
                    <td>{t_results["mean2"]:.4f}</td>
                </tr>
                <tr>
                    <td>Mean Difference</td>
                    <td>{t_results["mean_diff"]:.4f}</td>
                </tr>
                <tr>
                    <td>Cohen's d</td>
                    <td>{t_results["cohens_d"]:.4f}</td>
                </tr>
            </table>
            
            <p><strong>Interpretation:</strong> 
            {f"The p-value ({t_results['p_value']:.4f}) is less than 0.05, so we reject the null hypothesis. There is a statistically significant difference." 
              if t_results['p_value'] < 0.05 else 
             f"The p-value ({t_results['p_value']:.4f}) is greater than 0.05, so we cannot reject the null hypothesis. There is no statistically significant difference."}</p>
        """
    
    if "anova" in analysis_results:
        anova_results = analysis_results["anova"]["results"]
        html_content += f"""
            <h2>ANOVA Analysis</h2>
            <p>Comparing <strong>{analysis_results["anova"]["variable"]}</strong> across groups in <strong>{analysis_results["anova"]["groups_column"]}</strong></p>
            
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>F-statistic</td>
                    <td>{anova_results["f_statistic"]:.4f}</td>
                </tr>
                <tr>
                    <td>p-value</td>
                    <td>{anova_results["p_value"]:.4f}</td>
                </tr>
                <tr>
                    <td>Total Observations</td>
                    <td>{anova_results["total_observations"]}</td>
                </tr>
            </table>
            
            <p><strong>Interpretation:</strong> 
            {f"The p-value ({anova_results['p_value']:.4f}) is less than 0.05, so we reject the null hypothesis. There are statistically significant differences between groups." 
              if anova_results['p_value'] < 0.05 else 
             f"The p-value ({anova_results['p_value']:.4f}) is greater than 0.05, so we cannot reject the null hypothesis. There are no statistically significant differences between groups."}</p>
        """
    
    if "clustering" in analysis_results:
        html_content += f"""
            <h2>Clustering Analysis</h2>
            <p>Method: <strong>{analysis_results["clustering"]["method"]}</strong></p>
            <p>Number of clusters: <strong>{analysis_results["clustering"]["params"].get("n_clusters", "N/A")}</strong></p>
            
            <h3>Cluster Statistics</h3>
            {analysis_results["clustering"]["cluster_stats"].to_html()}
            
            <p>Columns used: {", ".join(analysis_results["clustering"]["columns_used"])}</p>
        """
    
    if "prediction" in analysis_results:
        html_content += f"""
            <h2>Prediction Model</h2>
            <p>Model: <strong>{analysis_results["prediction"]["model_type"]}</strong></p>
            <p>Target variable: <strong>{analysis_results["prediction"]["target"]}</strong></p>
            <p>Predictor variables: {", ".join(analysis_results["prediction"]["predictors"])}</p>
            
            <h3>Model Performance</h3>
            {analysis_results["prediction"]["metrics"].to_html(index=False)}
        """
        
        if analysis_results["prediction"]["feature_importance"] is not None:
            html_content += f"""
                <h3>Feature Importance</h3>
                {analysis_results["prediction"]["feature_importance"].to_html(index=False)}
            """
    
    # Close the HTML
    html_content += f"""
            <div class="footer">
                <p>Generated by BioData Explorer | {current_datetime}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML content to the file
    with open(report_path, "w") as f:
        f.write(html_content)
    
    return report_path
