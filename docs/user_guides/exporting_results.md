# Exporting Results Guide

This guide explains how to export and share your analysis results from BioData Explorer, ensuring your insights are preserved and can be effectively communicated to colleagues and included in publications.

## Export Types Overview

BioData Explorer offers several ways to export your work:

1. **Complete Analysis Reports**: Full documentation of your analysis workflow and results
2. **Individual Data Exports**: Raw data, processed data, and analysis outcomes
3. **Visualization Exports**: High-quality figures for publications and presentations
4. **Machine Learning Models**: Trained model exports for future use
5. **Workflow Documentation**: Step-by-step record of your analysis process

## Complete Analysis Reports

### HTML Reports

The most comprehensive export option that includes:
- Data summaries
- Statistical analyses
- Interactive visualizations
- Machine learning results
- Methodology documentation

#### How to Create an HTML Report:

1. Navigate to "Export Results" in the sidebar
2. Select "HTML Report" as the export format
3. Configure report options:
   - Report title
   - Include/exclude sections
   - Visualization quality settings
   - Toggle interactive elements
4. Click "Generate Report"
5. Use the Download button to save the HTML file

### Usage Tips for HTML Reports:

- HTML reports can be opened in any modern web browser
- Interactive elements remain functional when opened locally
- Reports can be shared via email or file sharing services
- For publication supplementary materials, consider providing both HTML and PDF versions

## Individual Data Exports

### CSV Exports

For exporting tabular data in a universal format:

1. Navigate to "Export Results" in the sidebar
2. Select "CSV Files" as the export format
3. Choose what to export:
   - Processed data
   - Statistical results
   - Clustering assignments
   - Model predictions
4. Click the corresponding "Export" button for each selection
5. Download the CSV files

### Excel Workbooks

For creating multi-sheet exports with related data:

1. Navigate to "Export Results" in the sidebar
2. Select "Excel Workbook" as the export format
3. Configure which sheets to include:
   - Data sheets
   - Analysis results
   - Statistical summaries
   - Model outputs
4. Click "Export to Excel"
5. Download the XLSX file

## Visualization Exports

### Individual Visualizations

For exporting specific plots for presentations or publications:

#### From Any Visualization:
1. Hover over the visualization
2. Use the toolbar in the top-right corner
3. Click the camera icon for PNG export
4. For more options, click the "Export" menu to select:
   - PNG: For general use
   - SVG: For publications (vector format)
   - PDF: For print-ready figures
   - HTML: For interactive sharing

### Visualization Customization Before Export

For publication-quality figures:
1. Adjust the visualization parameters:
   - Set appropriate axis labels
   - Add a descriptive title
   - Adjust color scales if needed
   - Add annotations for key features
2. Resize the plot using the width/height controls
3. Export using the highest quality setting

## Machine Learning Model Exports

For advanced users who want to save and reuse trained models:

1. Navigate to "Machine Learning" in the sidebar
2. After training a model, look for the "Export Model" option
3. Choose the export format:
   - Pickle file (for Python compatibility)
   - JSON (for cross-platform use)
4. Download the model file
5. To share, include the:
   - Model file
   - Feature names
   - Preprocessing steps

## Exporting for Publications

When preparing exports for academic publications:

### Figure Guidelines:
- Use SVG or high-resolution PNG (300+ DPI)
- Set appropriate dimensions for journal requirements
- Use color schemes suitable for both color and grayscale printing
- Include clear, descriptive captions in your manuscript

### Supplementary Data:
- Export raw data as CSV for data availability statements
- Provide analysis code or workflow documentation
- Include model parameters and evaluation metrics
- Use Excel for multi-tabbed supplementary data files

## Best Practices for Data Export

1. **Include Metadata**: Always document when the analysis was performed and which version of data was used
2. **Version Control**: Include date stamps or version numbers in filenames
3. **Data Dictionary**: Include a readme file or data dictionary explaining columns/variables
4. **Full Reproducibility**: Document software version and key parameters
5. **File Naming**: Use consistent, descriptive file naming conventions

## Sharing Results with Collaborators

For effective collaboration:

1. **Combined Reports**: Use HTML reports for comprehensive sharing
2. **Interactive Sharing**: When sharing visualizations, include interactive HTML versions
3. **Accessibility**: Ensure data formats are accessible to collaborators (consider CSV for universal access)
4. **Documentation**: Include methodology notes and parameter settings
5. **Data Size**: For large datasets, consider compression or sharing platforms with file size support

## Exporting for Long-term Storage

For archiving your analysis:

1. **Use Non-proprietary Formats**: CSV rather than Excel when possible
2. **Include Raw Data**: Always preserve the original, unmodified data
3. **Documentation**: Create a detailed methods document explaining the analysis
4. **File Organization**: Use logical folder structures for complex analyses
5. **Repository Submission**: Consider submitting to appropriate data repositories for your field

## Troubleshooting Exports

### Common Export Issues:

| Issue | Solution |
|-------|----------|
| Report generation takes too long | Reduce the number of visualizations or their complexity |
| Large file sizes | Use data sampling or focus on key results only |
| Missing visualizations in exports | Ensure all plots have rendered before export |
| Formatting issues in Excel | Check for special characters or data type mismatches |
| Interactive elements not working | Ensure you're using the HTML format and modern browser |

### Export Size Limits:

- HTML Reports: No strict limit, but browser performance may suffer over 20MB
- Excel Files: Best kept under 10MB for easy sharing
- CSV Files: No practical limit for export, but very large files may be slow to open

---

By following these guidelines, you can effectively export, preserve, and share your biological data analyses performed in BioData Explorer.