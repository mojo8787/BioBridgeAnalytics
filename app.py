import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import tempfile
import time
from components.data_loader import load_data, get_supported_formats
from components.data_processor import process_data, get_column_types, identify_potential_correlations
from components.visualization import create_scatter_plot, create_heatmap, create_box_plot, create_histogram, create_pca_plot
from components.statistical_analysis import perform_correlation_analysis, perform_anova, perform_ttest, perform_descriptive_stats
from components.ml_analysis import perform_clustering, perform_simple_ml_prediction
from components.file_handler import save_figure, export_analysis_report
from utils.helpers import format_bytes, get_data_summary
from utils.constants import EXAMPLE_DESCRIPTIONS, APP_INFO

# Import new API components
from components.genomic_api import search_ncbi_databases, get_sequence_record, fetch_protein_info_from_uniprot, search_uniprot, analyze_sequence
from components.ai_integration import analyze_text_with_openai, summarize_research_paper, extract_biological_entities, analyze_sequence_with_ai, huggingface_protein_prediction

# Set page config
st.set_page_config(
    page_title="BioData Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'file_size' not in st.session_state:
    st.session_state.file_size = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'column_types' not in st.session_state:
    st.session_state.column_types = None
if 'potential_correlations' not in st.session_state:
    st.session_state.potential_correlations = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'current_view' not in st.session_state:
    st.session_state.current_view = "upload"
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
    
# API related session state
if 'entrez_email' not in st.session_state:
    st.session_state.entrez_email = ""
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'sequence' not in st.session_state:
    st.session_state.sequence = ""
if 'ncbi_results' not in st.session_state:
    st.session_state.ncbi_results = None
if 'uniprot_results' not in st.session_state:
    st.session_state.uniprot_results = None
if 'ai_analysis_results' not in st.session_state:
    st.session_state.ai_analysis_results = None

# Main title
st.title("BioData Explorer 🧬")
st.subheader("A platform for exploring biological research datasets")

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    
    # Only show navigation options if data is loaded
    # Data Analysis section (only shown if data is loaded)
    if st.session_state.data is not None:
        st.sidebar.header("Data Analysis Tools")
        selected_page = st.radio(
            "Data Analysis:",
            ["Data Upload", "Data Overview", "Statistical Analysis", "Visualization", "Machine Learning", "Export Results"]
        )
        
        if selected_page == "Data Upload":
            st.session_state.current_view = "upload"
        elif selected_page == "Data Overview":
            st.session_state.current_view = "overview"
        elif selected_page == "Statistical Analysis":
            st.session_state.current_view = "statistics"
        elif selected_page == "Visualization":
            st.session_state.current_view = "visualization"
        elif selected_page == "Machine Learning":
            st.session_state.current_view = "ml"
        elif selected_page == "Export Results":
            st.session_state.current_view = "export"
    else:
        st.info("Upload a dataset to enable data analysis options")
    
    # API Tools section (available even without data upload)
    st.sidebar.header("API Tools")
    api_page = st.sidebar.radio(
        "API Tools:",
        ["Genomic/Protein API", "AI Integration"]
    )
    
    if api_page == "Genomic/Protein API":
        st.session_state.current_view = "genomic_api"
    elif api_page == "AI Integration":
        st.session_state.current_view = "ai_integration"
    
    # App information in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(APP_INFO)
    
    # Show supported formats
    st.sidebar.subheader("Supported Formats")
    formats = get_supported_formats()
    for fmt, desc in formats.items():
        st.sidebar.markdown(f"**{fmt}**: {desc}")

# Data upload view
if st.session_state.current_view == "upload" or st.session_state.data is None:
    st.header("Upload Your Biological Dataset")
    
    uploaded_file = st.file_uploader("Choose a file", 
                                     type=list(get_supported_formats().keys()),
                                     help="Upload a biological dataset file")
    
    if uploaded_file is not None:
        try:
            # Show a spinner while loading the data
            with st.spinner('Loading and processing your data...'):
                # Get file size
                file_size = uploaded_file.size
                st.session_state.file_size = format_bytes(file_size)
                st.session_state.filename = uploaded_file.name
                
                # Get file extension
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                st.session_state.file_type = file_extension
                
                # Load the data
                data = load_data(uploaded_file)
                
                if data is not None:
                    st.session_state.data = data
                    
                    # Process the data
                    st.session_state.processed_data = process_data(data)
                    
                    # Identify column types
                    st.session_state.column_types = get_column_types(st.session_state.processed_data)
                    
                    # Identify potential correlations
                    st.session_state.potential_correlations = identify_potential_correlations(st.session_state.processed_data)
                    
                    # Switch to overview view
                    st.session_state.current_view = "overview"
                    st.success("Data successfully loaded! Explore the data using the navigation menu.")
                    
                    # Add a small delay and rerun to show the overview
                    time.sleep(0.5)
                    st.rerun()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    # Information about the application
    st.markdown("---")
    st.subheader("About BioData Explorer")
    st.write("""
    BioData Explorer is a user-friendly platform designed specifically for biologists and
    researchers to analyze complex biological datasets without requiring extensive programming knowledge.
    You can upload various types of biological data files and get instant access to advanced
    statistical analyses, visualizations, and machine learning tools.
    """)
    
    # Example use cases
    st.subheader("Example Use Cases")
    for i, (title, desc) in enumerate(EXAMPLE_DESCRIPTIONS.items()):
        with st.expander(title):
            st.write(desc)

# Data overview view
elif st.session_state.current_view == "overview":
    st.header("Data Overview")
    
    # File information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Name", st.session_state.filename)
    with col2:
        st.metric("File Size", st.session_state.file_size)
    with col3:
        st.metric("File Type", st.session_state.file_type)
    
    # Data summary
    st.subheader("Dataset Summary")
    data_summary = get_data_summary(st.session_state.processed_data)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", data_summary["rows"])
        st.metric("Numeric Columns", data_summary["numeric_cols"])
    with col2:
        st.metric("Columns", data_summary["cols"])
        st.metric("Categorical Columns", data_summary["categorical_cols"])
    
    # Display first rows of the data
    st.subheader("Data Preview")
    st.dataframe(st.session_state.processed_data.head(10))
    
    # Column information
    st.subheader("Column Information")
    col_df = pd.DataFrame({
        "Column": st.session_state.column_types.keys(),
        "Type": [t["type"] for t in st.session_state.column_types.values()],
        "Missing Values": [t["missing_count"] for t in st.session_state.column_types.values()],
        "Missing (%)": [f"{t['missing_percentage']:.2f}%" for t in st.session_state.column_types.values()]
    })
    st.dataframe(col_df)
    
    # Potential correlations
    if st.session_state.potential_correlations:
        st.subheader("Potential Correlations")
        st.write("These column pairs show strong correlations that might be interesting for further analysis:")
        for pair, corr in st.session_state.potential_correlations:
            st.write(f"• **{pair[0]}** and **{pair[1]}**: correlation = {corr:.3f}")
    
    # Missing values handling
    st.subheader("Missing Values Handling")
    if st.checkbox("Show options for handling missing values"):
        missing_strategy = st.selectbox(
            "Select strategy for handling missing values:",
            ["Do nothing", "Drop rows with any missing values", "Fill numeric with mean", "Fill numeric with median", "Fill categorical with mode"]
        )
        
        if missing_strategy != "Do nothing" and st.button("Apply"):
            with st.spinner("Processing..."):
                if missing_strategy == "Drop rows with any missing values":
                    st.session_state.processed_data = st.session_state.processed_data.dropna()
                    st.success("Rows with missing values removed")
                elif missing_strategy == "Fill numeric with mean":
                    for col, info in st.session_state.column_types.items():
                        if info["type"] == "numeric" and info["missing_count"] > 0:
                            st.session_state.processed_data[col].fillna(st.session_state.processed_data[col].mean(), inplace=True)
                    st.success("Missing numeric values filled with mean")
                elif missing_strategy == "Fill numeric with median":
                    for col, info in st.session_state.column_types.items():
                        if info["type"] == "numeric" and info["missing_count"] > 0:
                            st.session_state.processed_data[col].fillna(st.session_state.processed_data[col].median(), inplace=True)
                    st.success("Missing numeric values filled with median")
                elif missing_strategy == "Fill categorical with mode":
                    for col, info in st.session_state.column_types.items():
                        if info["type"] == "categorical" and info["missing_count"] > 0:
                            st.session_state.processed_data[col].fillna(st.session_state.processed_data[col].mode()[0], inplace=True)
                    st.success("Missing categorical values filled with mode")
                
                # Update column types after handling missing values
                st.session_state.column_types = get_column_types(st.session_state.processed_data)
                st.rerun()

    # Data filtering options
    st.subheader("Data Filtering")
    if st.checkbox("Show filtering options"):
        col1, col2 = st.columns(2)
        
        with col1:
            filter_column = st.selectbox("Select column to filter on:", list(st.session_state.processed_data.columns))
        
        with col2:
            col_type = st.session_state.column_types[filter_column]["type"]
            if col_type == "numeric":
                min_val = float(st.session_state.processed_data[filter_column].min())
                max_val = float(st.session_state.processed_data[filter_column].max())
                filter_range = st.slider(f"Range for {filter_column}", 
                                         min_value=min_val, 
                                         max_value=max_val, 
                                         value=(min_val, max_val))
                filter_condition = (st.session_state.processed_data[filter_column] >= filter_range[0]) & \
                                   (st.session_state.processed_data[filter_column] <= filter_range[1])
            else:  # categorical
                unique_values = st.session_state.processed_data[filter_column].dropna().unique().tolist()
                selected_values = st.multiselect(f"Select values for {filter_column}", 
                                                 options=unique_values,
                                                 default=unique_values)
                filter_condition = st.session_state.processed_data[filter_column].isin(selected_values)
        
        if st.button("Apply Filter"):
            filtered_data = st.session_state.processed_data[filter_condition]
            st.session_state.processed_data = filtered_data
            st.success(f"Data filtered: {len(filtered_data)} rows remaining")
            st.rerun()

# Statistical Analysis View
elif st.session_state.current_view == "statistics":
    st.header("Statistical Analysis")
    
    # Select analysis type
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Descriptive Statistics", "Correlation Analysis", "T-Test", "ANOVA"]
    )
    
    if analysis_type == "Descriptive Statistics":
        st.subheader("Descriptive Statistics")
        
        # Select columns for analysis
        num_columns = [col for col, info in st.session_state.column_types.items() 
                       if info["type"] == "numeric"]
        
        selected_columns = st.multiselect(
            "Select columns for descriptive statistics:",
            options=num_columns,
            default=num_columns[:min(5, len(num_columns))]
        )
        
        if selected_columns:
            with st.spinner("Calculating statistics..."):
                stats_results = perform_descriptive_stats(
                    st.session_state.processed_data, selected_columns
                )
                
                st.dataframe(stats_results)
                
                # Save to session state for export
                st.session_state.analysis_results["descriptive_stats"] = stats_results
                
                # Visualize distributions
                if st.checkbox("Show distributions"):
                    for col in selected_columns:
                        st.subheader(f"Distribution of {col}")
                        fig = create_histogram(st.session_state.processed_data, col)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one column for analysis")
    
    elif analysis_type == "Correlation Analysis":
        st.subheader("Correlation Analysis")
        
        # Select columns for correlation
        num_columns = [col for col, info in st.session_state.column_types.items() 
                       if info["type"] == "numeric"]
        
        selected_columns = st.multiselect(
            "Select columns for correlation analysis:",
            options=num_columns,
            default=num_columns[:min(5, len(num_columns))]
        )
        
        if len(selected_columns) >= 2:
            correlation_method = st.radio(
                "Correlation method:",
                ["pearson", "spearman", "kendall"],
                index=0,
                help="Pearson (linear), Spearman (rank), Kendall (ordinal)"
            )
            
            with st.spinner("Calculating correlations..."):
                corr_matrix, p_values = perform_correlation_analysis(
                    st.session_state.processed_data, 
                    selected_columns,
                    method=correlation_method
                )
                
                # Display correlation matrix
                st.subheader("Correlation Matrix")
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None))
                
                # Display p-values
                st.subheader("P-Value Matrix")
                st.dataframe(p_values.style.highlight_between(left=0, right=0.05))
                
                # Visualize as heatmap
                st.subheader("Correlation Heatmap")
                fig = create_heatmap(corr_matrix)
                st.plotly_chart(fig, use_container_width=True)
                
                # Save to session state for export
                st.session_state.analysis_results["correlation"] = {
                    "matrix": corr_matrix,
                    "p_values": p_values,
                    "method": correlation_method
                }
                
                # Show strongest correlations
                st.subheader("Strongest Correlations")
                # Get the upper triangle of the correlation matrix
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                corr_pairs = []
                
                for i in range(len(selected_columns)):
                    for j in range(i+1, len(selected_columns)):
                        if not mask[i, j]:
                            corr_pairs.append({
                                'Variables': f"{selected_columns[i]} & {selected_columns[j]}",
                                'Correlation': corr_matrix.iloc[i, j],
                                'P-value': p_values.iloc[i, j]
                            })
                
                corr_pairs_df = pd.DataFrame(corr_pairs)
                corr_pairs_df = corr_pairs_df.sort_values('Correlation', key=abs, ascending=False)
                
                st.dataframe(corr_pairs_df)
                
                # Plot scatter for top correlations
                if not corr_pairs_df.empty:
                    top_pair = corr_pairs_df.iloc[0]['Variables'].split(' & ')
                    st.subheader(f"Scatter Plot for Top Correlation: {top_pair[0]} vs {top_pair[1]}")
                    fig = create_scatter_plot(
                        st.session_state.processed_data, 
                        x=top_pair[0], 
                        y=top_pair[1]
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least two columns for correlation analysis")
    
    elif analysis_type == "T-Test":
        st.subheader("T-Test Analysis")
        
        # Select numeric column for testing
        num_columns = [col for col, info in st.session_state.column_types.items() 
                       if info["type"] == "numeric"]
        
        numeric_column = st.selectbox(
            "Select numeric column for t-test:",
            options=num_columns
        )
        
        # Select categorical column for grouping
        cat_columns = [col for col, info in st.session_state.column_types.items() 
                       if info["type"] == "categorical"]
        
        if cat_columns:
            group_column = st.selectbox(
                "Select categorical column for grouping:",
                options=cat_columns
            )
            
            # Get unique values in the categorical column
            unique_groups = st.session_state.processed_data[group_column].dropna().unique().tolist()
            
            if len(unique_groups) >= 2:
                # Let user select two groups to compare
                col1, col2 = st.columns(2)
                with col1:
                    group1 = st.selectbox("Select first group:", options=unique_groups, index=0)
                with col2:
                    group2 = st.selectbox("Select second group:", options=unique_groups, index=min(1, len(unique_groups)-1))
                
                if group1 != group2:
                    equal_var = st.checkbox("Assume equal variance", value=True)
                    
                    with st.spinner("Performing t-test..."):
                        t_results = perform_ttest(
                            st.session_state.processed_data,
                            numeric_column,
                            group_column,
                            group1,
                            group2,
                            equal_var=equal_var
                        )
                        
                        # Display results
                        st.subheader("T-Test Results")
                        results_df = pd.DataFrame({
                            'Metric': ['t-statistic', 'p-value', 'Group 1 Mean', 'Group 2 Mean', 'Mean Difference'],
                            'Value': [
                                t_results['t_statistic'],
                                t_results['p_value'],
                                t_results['mean1'],
                                t_results['mean2'],
                                t_results['mean_diff']
                            ]
                        })
                        
                        st.dataframe(results_df)
                        
                        # Interpretation
                        st.subheader("Interpretation")
                        alpha = 0.05
                        if t_results['p_value'] < alpha:
                            st.info(f"The p-value ({t_results['p_value']:.4f}) is less than 0.05, so we reject the null hypothesis. "
                                   f"There is a statistically significant difference in {numeric_column} between the {group1} and {group2} groups.")
                        else:
                            st.info(f"The p-value ({t_results['p_value']:.4f}) is greater than 0.05, so we cannot reject the null hypothesis. "
                                   f"There is no statistically significant difference in {numeric_column} between the {group1} and {group2} groups.")
                        
                        # Visualization
                        st.subheader("Visual Comparison")
                        fig = create_box_plot(
                            st.session_state.processed_data,
                            numeric_column,
                            group_column,
                            [group1, group2]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Save to session state for export
                        st.session_state.analysis_results["ttest"] = {
                            "results": t_results,
                            "groups": [group1, group2],
                            "variable": numeric_column
                        }
                else:
                    st.warning("Please select two different groups to compare")
            else:
                st.warning(f"The selected categorical column '{group_column}' needs at least two unique values for t-test")
        else:
            st.warning("No categorical columns found in the dataset. T-test requires a categorical column for grouping.")
    
    elif analysis_type == "ANOVA":
        st.subheader("ANOVA Analysis")
        
        # Select numeric column for testing
        num_columns = [col for col, info in st.session_state.column_types.items() 
                      if info["type"] == "numeric"]
        
        numeric_column = st.selectbox(
            "Select numeric column for ANOVA:",
            options=num_columns
        )
        
        # Select categorical column for grouping
        cat_columns = [col for col, info in st.session_state.column_types.items() 
                      if info["type"] == "categorical"]
        
        if cat_columns:
            group_column = st.selectbox(
                "Select categorical column for grouping:",
                options=cat_columns
            )
            
            # Get unique values in the categorical column
            unique_groups = st.session_state.processed_data[group_column].dropna().unique().tolist()
            
            if len(unique_groups) >= 2:
                with st.spinner("Performing ANOVA..."):
                    anova_results = perform_anova(
                        st.session_state.processed_data,
                        numeric_column,
                        group_column
                    )
                    
                    # Display results
                    st.subheader("ANOVA Results")
                    results_df = pd.DataFrame({
                        'Metric': ['F-statistic', 'p-value', 'Groups', 'Total Observations'],
                        'Value': [
                            anova_results['f_statistic'],
                            anova_results['p_value'],
                            len(unique_groups),
                            anova_results['total_observations']
                        ]
                    })
                    
                    st.dataframe(results_df)
                    
                    # Interpretation
                    st.subheader("Interpretation")
                    alpha = 0.05
                    if anova_results['p_value'] < alpha:
                        st.info(f"The p-value ({anova_results['p_value']:.4f}) is less than 0.05, so we reject the null hypothesis. "
                               f"There are statistically significant differences in {numeric_column} among the groups in {group_column}.")
                    else:
                        st.info(f"The p-value ({anova_results['p_value']:.4f}) is greater than 0.05, so we cannot reject the null hypothesis. "
                               f"There are no statistically significant differences in {numeric_column} among the groups in {group_column}.")
                    
                    # Visualization
                    st.subheader("Visual Comparison")
                    fig = create_box_plot(
                        st.session_state.processed_data,
                        numeric_column,
                        group_column
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save to session state for export
                    st.session_state.analysis_results["anova"] = {
                        "results": anova_results,
                        "groups_column": group_column,
                        "variable": numeric_column
                    }
            else:
                st.warning(f"The selected categorical column '{group_column}' needs at least two unique values for ANOVA")
        else:
            st.warning("No categorical columns found in the dataset. ANOVA requires a categorical column for grouping.")

# Visualization View
elif st.session_state.current_view == "visualization":
    st.header("Data Visualization")
    
    # Select visualization type
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Scatter Plot", "Box Plot", "Histogram", "Heatmap", "PCA Plot"]
    )
    
    if viz_type == "Scatter Plot":
        st.subheader("Scatter Plot")
        
        # Select columns for scatter plot
        num_columns = [col for col, info in st.session_state.column_types.items() 
                       if info["type"] == "numeric"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_column = st.selectbox("X-axis:", options=num_columns, index=0)
        
        with col2:
            y_column = st.selectbox("Y-axis:", options=num_columns, index=min(1, len(num_columns)-1))
        
        with col3:
            # Optional color grouping
            color_options = ["None"] + [col for col, info in st.session_state.column_types.items() 
                                       if info["type"] == "categorical"]
            color_column = st.selectbox("Color by:", options=color_options)
            
            if color_column == "None":
                color_column = None
        
        # Create scatter plot
        fig = create_scatter_plot(
            st.session_state.processed_data,
            x=x_column,
            y=y_column,
            color=color_column
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Option to download the figure
        if st.button("Save Figure"):
            save_path = save_figure(fig, f"scatter_{x_column}_vs_{y_column}")
            with open(save_path, "rb") as f:
                btn = st.download_button(
                    label="Download figure",
                    data=f,
                    file_name=f"scatter_{x_column}_vs_{y_column}.html",
                    mime="text/html"
                )
    
    elif viz_type == "Box Plot":
        st.subheader("Box Plot")
        
        # Select numeric column for box plot
        num_columns = [col for col, info in st.session_state.column_types.items() 
                       if info["type"] == "numeric"]
        
        numeric_column = st.selectbox("Numeric variable:", options=num_columns)
        
        # Select categorical column for grouping
        cat_columns = [col for col, info in st.session_state.column_types.items() 
                       if info["type"] == "categorical"]
        
        if cat_columns:
            group_column = st.selectbox("Group by:", options=cat_columns)
            
            # Let user limit the number of groups if there are many
            unique_groups = st.session_state.processed_data[group_column].dropna().unique().tolist()
            if len(unique_groups) > 10:
                selected_groups = st.multiselect(
                    f"Select groups to display (max {len(unique_groups)} available):",
                    options=unique_groups,
                    default=unique_groups[:5]
                )
            else:
                selected_groups = unique_groups
            
            if selected_groups:
                # Create box plot
                fig = create_box_plot(
                    st.session_state.processed_data,
                    numeric_column,
                    group_column,
                    selected_groups
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Option to download the figure
                if st.button("Save Figure"):
                    save_path = save_figure(fig, f"boxplot_{numeric_column}_by_{group_column}")
                    with open(save_path, "rb") as f:
                        btn = st.download_button(
                            label="Download figure",
                            data=f,
                            file_name=f"boxplot_{numeric_column}_by_{group_column}.html",
                            mime="text/html"
                        )
            else:
                st.warning("Please select at least one group to display")
        else:
            st.warning("No categorical columns found in the dataset. Box plots require a categorical column for grouping.")
    
    elif viz_type == "Histogram":
        st.subheader("Histogram")
        
        # Select numeric column for histogram
        num_columns = [col for col, info in st.session_state.column_types.items() 
                       if info["type"] == "numeric"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            numeric_column = st.selectbox("Variable:", options=num_columns)
        
        with col2:
            bin_count = st.slider("Number of bins:", min_value=5, max_value=100, value=20)
        
        # Optional color grouping
        color_options = ["None"] + [col for col, info in st.session_state.column_types.items() 
                                   if info["type"] == "categorical"]
        color_column = st.selectbox("Group by (optional):", options=color_options)
        
        if color_column == "None":
            color_column = None
        
        # Create histogram
        fig = create_histogram(
            st.session_state.processed_data,
            numeric_column,
            bins=bin_count,
            color=color_column
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Option to download the figure
        if st.button("Save Figure"):
            save_path = save_figure(fig, f"histogram_{numeric_column}")
            with open(save_path, "rb") as f:
                btn = st.download_button(
                    label="Download figure",
                    data=f,
                    file_name=f"histogram_{numeric_column}.html",
                    mime="text/html"
                )
    
    elif viz_type == "Heatmap":
        st.subheader("Correlation Heatmap")
        
        # Select columns for heatmap
        num_columns = [col for col, info in st.session_state.column_types.items() 
                       if info["type"] == "numeric"]
        
        selected_columns = st.multiselect(
            "Select columns:",
            options=num_columns,
            default=num_columns[:min(8, len(num_columns))]
        )
        
        if len(selected_columns) >= 2:
            correlation_method = st.radio(
                "Correlation method:",
                ["pearson", "spearman", "kendall"],
                index=0
            )
            
            # Compute correlation matrix
            corr_matrix = st.session_state.processed_data[selected_columns].corr(method=correlation_method)
            
            # Create heatmap
            fig = create_heatmap(corr_matrix)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Option to download the figure
            if st.button("Save Figure"):
                save_path = save_figure(fig, f"heatmap_{correlation_method}")
                with open(save_path, "rb") as f:
                    btn = st.download_button(
                        label="Download figure",
                        data=f,
                        file_name=f"heatmap_{correlation_method}.html",
                        mime="text/html"
                    )
        else:
            st.warning("Please select at least two columns for the correlation heatmap")
    
    elif viz_type == "PCA Plot":
        st.subheader("Principal Component Analysis (PCA)")
        
        # Select columns for PCA
        num_columns = [col for col, info in st.session_state.column_types.items() 
                       if info["type"] == "numeric"]
        
        selected_columns = st.multiselect(
            "Select columns for PCA:",
            options=num_columns,
            default=num_columns[:min(5, len(num_columns))]
        )
        
        if len(selected_columns) >= 2:
            # Optional color grouping
            color_options = ["None"] + [col for col, info in st.session_state.column_types.items() 
                                       if info["type"] == "categorical"]
            color_column = st.selectbox("Color by:", options=color_options)
            
            if color_column == "None":
                color_column = None
            
            # Number of components
            n_components = st.slider(
                "Number of components to calculate:",
                min_value=2,
                max_value=min(len(selected_columns), 10),
                value=2
            )
            
            # Standardize data
            standardize = st.checkbox("Standardize data", value=True)
            
            with st.spinner("Calculating PCA..."):
                # Create PCA plot
                fig, explained_variance, loadings = create_pca_plot(
                    st.session_state.processed_data,
                    selected_columns,
                    n_components=n_components,
                    color=color_column,
                    standardize=standardize
                )
                
                # Display PCA plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Display explained variance
                st.subheader("Explained Variance")
                explained_var_df = pd.DataFrame({
                    'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance))],
                    'Explained Variance Ratio': explained_variance,
                    'Cumulative Variance Ratio': np.cumsum(explained_variance)
                })
                
                st.dataframe(explained_var_df)
                
                # Display feature loadings
                st.subheader("Feature Loadings")
                st.dataframe(loadings)
                
                # Option to download the figure
                if st.button("Save Figure"):
                    save_path = save_figure(fig, "pca_plot")
                    with open(save_path, "rb") as f:
                        btn = st.download_button(
                            label="Download figure",
                            data=f,
                            file_name="pca_plot.html",
                            mime="text/html"
                        )
        else:
            st.warning("Please select at least two columns for PCA")

# Machine Learning View
elif st.session_state.current_view == "ml":
    st.header("Machine Learning Analysis")
    
    # Select ML analysis type
    ml_type = st.selectbox(
        "Select Analysis Type",
        ["Clustering", "Simple Prediction"]
    )
    
    if ml_type == "Clustering":
        st.subheader("Data Clustering")
        
        # Select columns for clustering
        num_columns = [col for col, info in st.session_state.column_types.items() 
                       if info["type"] == "numeric"]
        
        selected_columns = st.multiselect(
            "Select columns for clustering:",
            options=num_columns,
            default=num_columns[:min(3, len(num_columns))]
        )
        
        if len(selected_columns) >= 2:
            # Clustering method
            method = st.selectbox(
                "Clustering method:",
                ["K-Means", "Hierarchical", "DBSCAN"]
            )
            
            # Clustering parameters
            if method == "K-Means":
                n_clusters = st.slider(
                    "Number of clusters:",
                    min_value=2,
                    max_value=10,
                    value=3
                )
                
                params = {
                    "n_clusters": n_clusters
                }
            
            elif method == "Hierarchical":
                n_clusters = st.slider(
                    "Number of clusters:",
                    min_value=2,
                    max_value=10,
                    value=3
                )
                
                linkage = st.selectbox(
                    "Linkage method:",
                    ["ward", "complete", "average", "single"]
                )
                
                params = {
                    "n_clusters": n_clusters,
                    "linkage": linkage
                }
            
            elif method == "DBSCAN":
                eps = st.slider(
                    "Epsilon (neighborhood distance):",
                    min_value=0.1,
                    max_value=5.0,
                    value=0.5,
                    step=0.1
                )
                
                min_samples = st.slider(
                    "Min samples:",
                    min_value=2,
                    max_value=20,
                    value=5
                )
                
                params = {
                    "eps": eps,
                    "min_samples": min_samples
                }
            
            # Standardize data
            standardize = st.checkbox("Standardize data", value=True)
            
            if st.button("Perform Clustering"):
                with st.spinner("Clustering data..."):
                    # Perform clustering
                    results = perform_clustering(
                        st.session_state.processed_data,
                        selected_columns,
                        method=method,
                        params=params,
                        standardize=standardize
                    )
                    
                    # Add cluster labels to data
                    data_with_clusters = st.session_state.processed_data.copy()
                    data_with_clusters['Cluster'] = results['labels']
                    
                    # Display cluster statistics
                    st.subheader("Cluster Statistics")
                    
                    cluster_stats = data_with_clusters.groupby('Cluster').agg({
                        column: ['mean', 'std', 'min', 'max', 'count'] for column in selected_columns
                    })
                    
                    st.dataframe(cluster_stats)
                    
                    # Display cluster visualization
                    st.subheader("Cluster Visualization")
                    
                    if len(selected_columns) >= 2:
                        if len(selected_columns) == 2:
                            # 2D scatter plot
                            fig = create_scatter_plot(
                                data_with_clusters,
                                x=selected_columns[0],
                                y=selected_columns[1],
                                color='Cluster'
                            )
                        else:
                            # PCA plot for more than 2 dimensions
                            fig, _, _ = create_pca_plot(
                                data_with_clusters,
                                selected_columns,
                                n_components=2,
                                color='Cluster',
                                standardize=standardize
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Option to add cluster labels to the dataset
                    if st.checkbox("Add cluster labels to the dataset"):
                        st.session_state.processed_data = data_with_clusters
                        st.session_state.column_types['Cluster'] = {
                            'type': 'categorical',
                            'missing_count': 0,
                            'missing_percentage': 0.0
                        }
                        st.success("Cluster labels added to the dataset")
                    
                    # Save to session state for export
                    st.session_state.analysis_results["clustering"] = {
                        "method": method,
                        "params": params,
                        "cluster_stats": cluster_stats,
                        "columns_used": selected_columns
                    }
        else:
            st.warning("Please select at least two columns for clustering")
    
    elif ml_type == "Simple Prediction":
        st.subheader("Simple Prediction Model")
        
        # Select target variable
        num_columns = [col for col, info in st.session_state.column_types.items() 
                       if info["type"] == "numeric"]
        
        target_column = st.selectbox(
            "Select target variable to predict:",
            options=num_columns
        )
        
        # Select predictor variables
        predictor_options = [col for col in num_columns if col != target_column]
        
        selected_predictors = st.multiselect(
            "Select predictor variables:",
            options=predictor_options,
            default=predictor_options[:min(3, len(predictor_options))]
        )
        
        if selected_predictors:
            # Model type
            model_type = st.selectbox(
                "Model type:",
                ["Linear Regression", "Random Forest", "Support Vector Machine"]
            )
            
            # Train-test split
            test_size = st.slider(
                "Test set size (%):",
                min_value=10,
                max_value=50,
                value=20
            ) / 100
            
            # Random state for reproducibility
            random_state = st.number_input(
                "Random seed:",
                min_value=1,
                max_value=1000,
                value=42
            )
            
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    # Train prediction model
                    results = perform_simple_ml_prediction(
                        st.session_state.processed_data,
                        target_column,
                        selected_predictors,
                        model_type=model_type,
                        test_size=test_size,
                        random_state=random_state
                    )
                    
                    # Display model performance
                    st.subheader("Model Performance")
                    
                    metrics_df = pd.DataFrame({
                        'Metric': ['R² (Train)', 'R² (Test)', 'MAE', 'MSE', 'RMSE'],
                        'Value': [
                            results['r2_train'],
                            results['r2_test'],
                            results['mae'],
                            results['mse'],
                            results['rmse']
                        ]
                    })
                    
                    st.dataframe(metrics_df)
                    
                    # Feature importance
                    if model_type in ["Linear Regression", "Random Forest"]:
                        st.subheader("Feature Importance")
                        
                        importance_df = pd.DataFrame({
                            'Feature': selected_predictors,
                            'Importance': results['feature_importance']
                        }).sort_values('Importance', ascending=False)
                        
                        st.dataframe(importance_df)
                        
                        # Visualize feature importance
                        fig = create_histogram(
                            importance_df,
                            x='Feature',
                            y='Importance',
                            title='Feature Importance'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Predicted vs Actual scatter plot
                    st.subheader("Predicted vs Actual Values")
                    
                    # Create DataFrame with actual and predicted values
                    prediction_df = pd.DataFrame({
                        'Actual': results['y_test'],
                        'Predicted': results['y_pred']
                    })
                    
                    # Create scatter plot
                    fig = create_scatter_plot(
                        prediction_df,
                        x='Actual',
                        y='Predicted',
                        title='Predicted vs Actual Values'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save to session state for export
                    st.session_state.analysis_results["prediction"] = {
                        "model_type": model_type,
                        "target": target_column,
                        "predictors": selected_predictors,
                        "metrics": metrics_df,
                        "feature_importance": importance_df if model_type in ["Linear Regression", "Random Forest"] else None
                    }
        else:
            st.warning("Please select at least one predictor variable")

# Export Results View
elif st.session_state.current_view == "export":
    st.header("Export Results")
    
    # Check if there are results to export
    if not st.session_state.analysis_results:
        st.warning("No analysis results to export yet. Perform some analyses first.")
    else:
        export_format = st.radio(
            "Export format:",
            ["HTML Report", "CSV Files", "Excel Workbook"]
        )
        
        if export_format == "HTML Report":
            report_title = st.text_input("Report title:", "BioData Analysis Report")
            include_visualizations = st.checkbox("Include visualizations", value=True)
            
            if st.button("Generate Report"):
                with st.spinner("Generating HTML report..."):
                    # Generate HTML report
                    report_path = export_analysis_report(
                        st.session_state.processed_data,
                        st.session_state.analysis_results,
                        report_title=report_title,
                        include_visualizations=include_visualizations
                    )
                    
                    # Create download button
                    with open(report_path, "rb") as f:
                        report_data = f.read()
                        
                        st.download_button(
                            label="Download HTML Report",
                            data=report_data,
                            file_name="biodata_analysis_report.html",
                            mime="text/html"
                        )
        
        elif export_format == "CSV Files":
            # Export processed data
            if st.button("Export Processed Data"):
                with st.spinner("Preparing CSV export..."):
                    # Convert to CSV
                    csv_buffer = io.StringIO()
                    st.session_state.processed_data.to_csv(csv_buffer, index=False)
                    
                    # Create download button
                    st.download_button(
                        label="Download Processed Data (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name="processed_data.csv",
                        mime="text/csv"
                    )
            
            # Export results if available
            if "descriptive_stats" in st.session_state.analysis_results:
                if st.button("Export Descriptive Statistics"):
                    stats_csv = st.session_state.analysis_results["descriptive_stats"].to_csv(index=True)
                    st.download_button(
                        label="Download Descriptive Statistics (CSV)",
                        data=stats_csv,
                        file_name="descriptive_statistics.csv",
                        mime="text/csv"
                    )
            
            if "correlation" in st.session_state.analysis_results:
                if st.button("Export Correlation Matrix"):
                    corr_csv = st.session_state.analysis_results["correlation"]["matrix"].to_csv(index=True)
                    st.download_button(
                        label="Download Correlation Matrix (CSV)",
                        data=corr_csv,
                        file_name="correlation_matrix.csv",
                        mime="text/csv"
                    )
        
        elif export_format == "Excel Workbook":
            if st.button("Export to Excel"):
                with st.spinner("Preparing Excel export..."):
                    # Create Excel file in memory
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        # Write processed data
                        st.session_state.processed_data.to_excel(writer, sheet_name="Data", index=False)
                        
                        # Write descriptive stats if available
                        if "descriptive_stats" in st.session_state.analysis_results:
                            st.session_state.analysis_results["descriptive_stats"].to_excel(
                                writer, sheet_name="Descriptive_Stats"
                            )
                        
                        # Write correlation matrix if available
                        if "correlation" in st.session_state.analysis_results:
                            st.session_state.analysis_results["correlation"]["matrix"].to_excel(
                                writer, sheet_name="Correlation_Matrix"
                            )
                        
                        # Write clustering results if available
                        if "clustering" in st.session_state.analysis_results:
                            if "cluster_stats" in st.session_state.analysis_results["clustering"]:
                                st.session_state.analysis_results["clustering"]["cluster_stats"].to_excel(
                                    writer, sheet_name="Cluster_Statistics"
                                )
                        
                        # Write prediction results if available
                        if "prediction" in st.session_state.analysis_results:
                            if "metrics" in st.session_state.analysis_results["prediction"]:
                                st.session_state.analysis_results["prediction"]["metrics"].to_excel(
                                    writer, sheet_name="Prediction_Metrics"
                                )
                            
                            if "feature_importance" in st.session_state.analysis_results["prediction"] and \
                               st.session_state.analysis_results["prediction"]["feature_importance"] is not None:
                                st.session_state.analysis_results["prediction"]["feature_importance"].to_excel(
                                    writer, sheet_name="Feature_Importance"
                                )
                    
                    excel_data = output.getvalue()
                    
                    # Create download button
                    st.download_button(
                        label="Download Excel Workbook",
                        data=excel_data,
                        file_name="biodata_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

# Genomic/Protein API View
elif st.session_state.current_view == "genomic_api":
    st.header("Genomic & Protein API Tools")
    st.write("""
    Access and analyze biological sequence data from public databases like NCBI and UniProt.
    These tools help you search for genes and proteins, retrieve detailed information,
    and perform basic sequence analysis.
    """)
    
    # API Configuration section
    with st.expander("API Configuration", expanded=True):
        st.write("Configure your API settings for genomic and protein databases.")
        
        # Email for NCBI Entrez
        entrez_email = st.text_input(
            "Email for NCBI Entrez API",
            value=st.session_state.entrez_email,
            help="Required for NCBI database access. Used to identify your requests to NCBI."
        )
        
        if entrez_email != st.session_state.entrez_email:
            st.session_state.entrez_email = entrez_email
            # Update the Entrez email in the genomic_api module
            from components.genomic_api import Entrez
            Entrez.email = entrez_email
    
    # Create tabs for different API functions
    api_tab = st.tabs(["NCBI Search", "UniProt Search", "Sequence Analysis"])
    
    # NCBI Search Tab
    with api_tab[0]:
        st.subheader("Search NCBI Databases")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            ncbi_search_term = st.text_input(
                "Search Term",
                placeholder="e.g., BRCA1, insulin, Escherichia coli",
                help="Enter gene, protein, organism, or keyword"
            )
        
        with col2:
            ncbi_database = st.selectbox(
                "Database",
                ["nucleotide", "protein", "gene", "pubmed"],
                help="Select which NCBI database to search"
            )
        
        with col3:
            ncbi_max_results = st.number_input(
                "Max Results",
                min_value=1,
                max_value=100,
                value=10,
                help="Maximum number of results to retrieve"
            )
        
        search_button = st.button("Search NCBI", use_container_width=True)
        
        if search_button and ncbi_search_term:
            if not entrez_email:
                st.error("Please provide an email address for NCBI Entrez API")
            else:
                with st.spinner("Searching NCBI databases..."):
                    try:
                        results = search_ncbi_databases(
                            term=ncbi_search_term,
                            database=ncbi_database,
                            max_results=ncbi_max_results
                        )
                        
                        if results and results.get("id_list"):
                            st.session_state.ncbi_results = results
                            st.success(f"Found {len(results['id_list'])} results")
                            
                            # Display the IDs
                            st.subheader("Result IDs")
                            st.write("Click on an ID to view detailed information")
                            
                            # Create buttons for each ID
                            cols = st.columns(3)
                            for i, id in enumerate(results["id_list"]):
                                if cols[i % 3].button(id, key=f"ncbi_id_{id}"):
                                    with st.spinner(f"Fetching details for {id}..."):
                                        details = get_sequence_record(id, database=ncbi_database)
                                        if details:
                                            st.subheader(f"Details for {id}")
                                            
                                            # Display basic info
                                            st.markdown(f"**Name:** {details['name']}")
                                            st.markdown(f"**Description:** {details['description']}")
                                            st.markdown(f"**Length:** {details['length']} bp/aa")
                                            
                                            # Sequence with expander
                                            with st.expander("Sequence"):
                                                st.text_area("", value=details['sequence'], height=200)
                                                
                                            # Features in a dataframe
                                            if details['features']:
                                                st.subheader("Features")
                                                features_df = pd.DataFrame([
                                                    {
                                                        'Type': f['type'],
                                                        'Location': f['location'],
                                                        'Qualifiers': ', '.join([f"{k}: {v if not isinstance(v, list) else ', '.join(v)}" 
                                                                              for k, v in f['qualifiers'].items()])
                                                    }
                                                    for f in details['features'][:20]  # Limit to first 20 features
                                                ])
                                                st.dataframe(features_df)
                                                
                                            # Annotations
                                            with st.expander("Annotations"):
                                                for key, value in details['annotations'].items():
                                                    if isinstance(value, (str, int, float)):
                                                        st.markdown(f"**{key}:** {value}")
                                                    elif isinstance(value, list) and len(value) < 5:
                                                        st.markdown(f"**{key}:** {', '.join(str(v) for v in value)}")
                                                    else:
                                                        st.markdown(f"**{key}:** *Complex data*")
                                        else:
                                            st.error(f"Could not fetch details for {id}")
                        else:
                            st.warning("No results found. Try a different search term.")
                    except Exception as e:
                        st.error(f"Error searching NCBI: {str(e)}")
    
    # UniProt Search Tab
    with api_tab[1]:
        st.subheader("Search UniProt Database")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uniprot_query = st.text_input(
                "Search Query",
                placeholder="e.g., insulin human, P01308, hemoglobin",
                help="Enter protein name, UniProt ID, gene name, or organism"
            )
        
        with col2:
            uniprot_limit = st.number_input(
                "Result Limit",
                min_value=1,
                max_value=50,
                value=10,
                help="Maximum number of results to retrieve"
            )
        
        search_uniprot_button = st.button("Search UniProt", use_container_width=True)
        
        if search_uniprot_button and uniprot_query:
            with st.spinner("Searching UniProt database..."):
                try:
                    results = search_uniprot(uniprot_query, limit=uniprot_limit)
                    
                    if results:
                        st.session_state.uniprot_results = results
                        st.success(f"Found {len(results)} results")
                        
                        # Create a dataframe of results
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df)
                        
                        # Allow selecting an entry to view details
                        selected_id = st.selectbox(
                            "Select an entry to view details:",
                            [f"{r['id']} - {r['protein_name']}" for r in results]
                        )
                        
                        if selected_id:
                            uniprot_id = selected_id.split(" - ")[0]
                            with st.spinner(f"Fetching details for {uniprot_id}..."):
                                protein_info = fetch_protein_info_from_uniprot(uniprot_id)
                                
                                if protein_info:
                                    st.subheader(f"Protein Details: {protein_info['name']}")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown(f"**ID:** {protein_info['id']}")
                                        st.markdown(f"**Gene:** {protein_info['gene']}")
                                        st.markdown(f"**Organism:** {protein_info['organism']}")
                                        st.markdown(f"**Length:** {protein_info['length']} amino acids")
                                    
                                    with col2:
                                        if protein_info['function']:
                                            st.markdown("**Function:**")
                                            st.markdown(f"_{protein_info['function']}_")
                                    
                                    # Sequence with expander
                                    with st.expander("Protein Sequence"):
                                        sequence = protein_info['sequence']
                                        st.text_area("", value=sequence, height=200)
                                        
                                        # Copy button
                                        if st.button("Copy Sequence to Clipboard"):
                                            st.session_state.sequence = sequence
                                            st.success("Sequence copied to clipboard and available in the Sequence Analysis tab")
                                    
                                    # Features
                                    if protein_info['features'] and len(protein_info['features']) > 0:
                                        st.subheader("Features")
                                        
                                        # Create a simpler representation for the dataframe
                                        features_simple = []
                                        for feature in protein_info['features']:
                                            feature_simple = {
                                                'Type': feature['type'],
                                                'Description': feature['description']
                                            }
                                            
                                            # Extract location information
                                            if isinstance(feature['location'], dict):
                                                if 'start' in feature['location'] and 'end' in feature['location']:
                                                    start = feature['location'].get('start', {}).get('value', '?')
                                                    end = feature['location'].get('end', {}).get('value', '?')
                                                    feature_simple['Position'] = f"{start}-{end}"
                                                elif 'position' in feature['location']:
                                                    pos = feature['location'].get('position', {}).get('value', '?')
                                                    feature_simple['Position'] = pos
                                                else:
                                                    feature_simple['Position'] = "Unknown"
                                            else:
                                                feature_simple['Position'] = "Unknown"
                                                
                                            features_simple.append(feature_simple)
                                        
                                        features_df = pd.DataFrame(features_simple)
                                        st.dataframe(features_df)
                                else:
                                    st.error(f"Could not fetch details for {uniprot_id}")
                    else:
                        st.warning("No results found. Try a different search query.")
                except Exception as e:
                    st.error(f"Error searching UniProt: {str(e)}")
    
    # Sequence Analysis Tab
    with api_tab[2]:
        st.subheader("Sequence Analysis")
        st.write("Analyze DNA, RNA, or protein sequences using bioinformatics tools.")
        
        # Input sequence
        sequence_input = st.text_area(
            "Enter Sequence",
            value=st.session_state.sequence,
            height=200,
            placeholder="Paste your DNA, RNA, or protein sequence here...",
            help="Input raw sequence without headers or formatting"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Allow user to clean the sequence
            clean_sequence = st.checkbox(
                "Clean sequence (remove numbers, spaces, and non-standard characters)",
                value=True
            )
        
        with col2:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["basic", "detailed"]
            )
        
        # Analyze button
        analyze_button = st.button("Analyze Sequence", use_container_width=True)
        
        if analyze_button and sequence_input:
            # Clean the sequence if requested
            if clean_sequence:
                cleaned_sequence = ''.join(c for c in sequence_input if c.isalpha())
                if cleaned_sequence != sequence_input:
                    st.info(f"Cleaned sequence from {len(sequence_input)} to {len(cleaned_sequence)} characters")
                    sequence_input = cleaned_sequence
            
            # Save to session state
            st.session_state.sequence = sequence_input
            
            with st.spinner("Analyzing sequence..."):
                try:
                    results = analyze_sequence(sequence_input, analysis_type=analysis_type)
                    
                    if results:
                        st.success("Analysis complete!")
                        
                        # Basic information
                        st.subheader("Basic Information")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Length", results["length"], help="Number of nucleotides or amino acids")
                        col2.metric("Molecular Weight", f"{results['molecular_weight']:.2f}", help="Molecular weight in Da")
                        
                        if "is_dna" in results:
                            col3.metric("Sequence Type", "DNA/RNA" if results["is_dna"] else "Protein")
                            
                            if results["is_dna"]:
                                col3.metric("GC Content", f"{results['gc_content']:.2f}%", help="Percentage of G and C nucleotides")
                        
                        # DNA-specific analysis
                        if "is_dna" in results and results["is_dna"]:
                            st.subheader("DNA Analysis")
                            
                            # Nucleotide composition
                            st.write("Nucleotide Composition:")
                            nucleotide_df = pd.DataFrame({
                                'Nucleotide': results["nucleotide_counts"].keys(),
                                'Count': results["nucleotide_counts"].values(),
                                'Percentage': [f"{(count/results['length'])*100:.2f}%" 
                                             for count in results["nucleotide_counts"].values()]
                            })
                            st.dataframe(nucleotide_df)
                            
                            # Transcription and translation
                            st.write("Transcription (DNA → RNA):")
                            st.text_area("RNA", results["transcription"], height=100)
                            
                            st.write("Translation (RNA → Protein):")
                            st.text_area("Protein", results["translation"], height=100)
                        
                        # Protein-specific analysis
                        elif "is_dna" in results and not results["is_dna"]:
                            st.subheader("Protein Analysis")
                            
                            # Create tabs for different protein analyses
                            protein_tabs = st.tabs(["Amino Acid Composition", "Physicochemical Properties", "Structure Prediction"])
                            
                            # Amino Acid Composition tab
                            with protein_tabs[0]:
                                if "amino_acid_counts" in results:
                                    aa_counts = results["amino_acid_counts"]
                                    aa_df = pd.DataFrame({
                                        'Amino Acid': aa_counts.keys(),
                                        'Count': aa_counts.values(),
                                        'Percentage': [f"{(count/results['length'])*100:.2f}%" 
                                                     for count in aa_counts.values()]
                                    })
                                    st.dataframe(aa_df)
                            
                            # Physicochemical Properties tab
                            with protein_tabs[1]:
                                if "isoelectric_point" in results:
                                    col1, col2 = st.columns(2)
                                    col1.metric("Isoelectric Point (pI)", f"{results['isoelectric_point']:.2f}")
                                    col2.metric("Hydrophobicity (GRAVY)", f"{results['gravy']:.3f}")
                                    
                                    col1, col2 = st.columns(2)
                                    col1.metric("Aromaticity", f"{results['aromaticity']:.3f}")
                                    col2.metric("Instability Index", f"{results['instability_index']:.2f}", 
                                              help="Value > 40 suggests unstable protein")
                            
                            # Structure Prediction tab
                            with protein_tabs[2]:
                                if "secondary_structure_fraction" in results:
                                    ss_fraction = results["secondary_structure_fraction"]
                                    st.write("Predicted Secondary Structure Composition:")
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("Helix", f"{ss_fraction[0]:.2%}")
                                    col2.metric("Sheet", f"{ss_fraction[1]:.2%}")
                                    col3.metric("Coil", f"{ss_fraction[2]:.2%}")
                    else:
                        st.error("Could not analyze the sequence. Please check if it's a valid biological sequence.")
                except Exception as e:
                    st.error(f"Error analyzing sequence: {str(e)}")

# AI Integration View
elif st.session_state.current_view == "ai_integration":
    st.header("AI Integration Tools")
    st.write("""
    Leverage artificial intelligence to analyze biological data, summarize research papers,
    and gain insights from complex biological information.
    """)
    
    # API Configuration section
    with st.expander("API Configuration", expanded=True):
        st.write("Configure your API settings for AI services.")
        
        # OpenAI API Key
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            help="Required for OpenAI services. Get your API key from openai.com"
        )
        
        if openai_api_key != st.session_state.openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            # Set the API key in the OpenAI package
            import openai
            openai.api_key = openai_api_key
    
    # Create tabs for different AI functions
    ai_tab = st.tabs([
        "Research Paper Analysis", 
        "Biological Text Mining", 
        "Sequence Analysis with AI", 
        "Protein Structure Prediction"
    ])
    
    # Research Paper Analysis Tab
    with ai_tab[0]:
        st.subheader("Research Paper Analysis")
        st.write("Upload or paste a research paper to get an AI-generated summary and key findings.")
        
        # Text input method
        input_method = st.radio(
            "Input Method",
            ["Paste Text", "Upload PDF (Coming Soon)"]
        )
        
        if input_method == "Paste Text":
            paper_text = st.text_area(
                "Paste Research Paper Text",
                height=300,
                placeholder="Paste the abstract or full text of a research paper here..."
            )
            
            # Focus area
            focus_area = st.text_input(
                "Focus Area (Optional)",
                placeholder="e.g., gene expression, protein folding, clinical implications",
                help="Specify a particular aspect you want the summary to focus on"
            )
            
            # Analyze button
            analyze_paper_button = st.button("Analyze Paper", use_container_width=True)
            
            if analyze_paper_button and paper_text:
                if not openai_api_key:
                    st.error("Please provide an OpenAI API key in the API Configuration section")
                else:
                    with st.spinner("Analyzing research paper..."):
                        try:
                            summary = summarize_research_paper(
                                text=paper_text,
                                focus_area=focus_area if focus_area else None
                            )
                            
                            if summary:
                                st.success("Analysis complete!")
                                st.subheader("Research Paper Summary")
                                st.markdown(summary)
                            else:
                                st.error("Could not generate summary. Please check your API key and try again.")
                        except Exception as e:
                            st.error(f"Error analyzing research paper: {str(e)}")
        else:
            st.info("PDF upload functionality will be available in a future update.")
    
    # Biological Text Mining Tab
    with ai_tab[1]:
        st.subheader("Biological Text Mining")
        st.write("Extract biological entities and relationships from text using AI.")
        
        bio_text = st.text_area(
            "Enter Biological Text",
            height=250,
            placeholder="Paste text containing biological entities (e.g., genes, proteins, diseases, compounds)..."
        )
        
        extract_button = st.button("Extract Entities", use_container_width=True)
        
        if extract_button and bio_text:
            if not openai_api_key:
                st.error("Please provide an OpenAI API key in the API Configuration section")
            else:
                with st.spinner("Extracting biological entities..."):
                    try:
                        entities = extract_biological_entities(bio_text)
                        
                        if entities:
                            st.success("Extraction complete!")
                            
                            # Create tabs for different entity types
                            entity_tabs = st.tabs([
                                "Genes & Proteins", 
                                "Organisms", 
                                "Diseases", 
                                "Chemicals", 
                                "Processes", 
                                "Techniques"
                            ])
                            
                            # Display entities by category
                            with entity_tabs[0]:
                                if "Genes and proteins" in entities and entities["Genes and proteins"]:
                                    st.write(", ".join(entities["Genes and proteins"]))
                                else:
                                    st.info("No genes or proteins identified.")
                                    
                            with entity_tabs[1]:
                                if "Organisms and species" in entities and entities["Organisms and species"]:
                                    st.write(", ".join(entities["Organisms and species"]))
                                else:
                                    st.info("No organisms identified.")
                                    
                            with entity_tabs[2]:
                                if "Diseases and conditions" in entities and entities["Diseases and conditions"]:
                                    st.write(", ".join(entities["Diseases and conditions"]))
                                else:
                                    st.info("No diseases identified.")
                                    
                            with entity_tabs[3]:
                                if "Chemical compounds and drugs" in entities and entities["Chemical compounds and drugs"]:
                                    st.write(", ".join(entities["Chemical compounds and drugs"]))
                                else:
                                    st.info("No chemicals identified.")
                                    
                            with entity_tabs[4]:
                                if "Biological processes" in entities and entities["Biological processes"]:
                                    st.write(", ".join(entities["Biological processes"]))
                                else:
                                    st.info("No biological processes identified.")
                                    
                            with entity_tabs[5]:
                                if "Laboratory techniques" in entities and entities["Laboratory techniques"]:
                                    st.write(", ".join(entities["Laboratory techniques"]))
                                else:
                                    st.info("No laboratory techniques identified.")
                        else:
                            st.error("Could not extract entities. Please check your API key and try again.")
                    except Exception as e:
                        st.error(f"Error extracting entities: {str(e)}")
    
    # Sequence Analysis with AI Tab
    with ai_tab[2]:
        st.subheader("Sequence Analysis with AI")
        st.write("Analyze biological sequences using AI to predict functions and properties.")
        
        # Input sequence
        ai_sequence_input = st.text_area(
            "Enter Biological Sequence",
            height=200,
            placeholder="Paste a DNA, RNA, or protein sequence here...",
            help="Input raw sequence without headers or formatting"
        )
        
        # Analysis type
        ai_analysis_type = st.selectbox(
            "Analysis Type",
            ["Function Prediction", "Structural Prediction"],
            help="Select the type of AI analysis to perform"
        )
        
        analyze_with_ai_button = st.button("Analyze with AI", use_container_width=True)
        
        if analyze_with_ai_button and ai_sequence_input:
            if not openai_api_key:
                st.error("Please provide an OpenAI API key in the API Configuration section")
            else:
                # Clean the sequence
                cleaned_sequence = ''.join(c for c in ai_sequence_input if c.isalpha())
                
                with st.spinner("Analyzing sequence with AI..."):
                    try:
                        analysis_type_param = "function_prediction"
                        if ai_analysis_type == "Structural Prediction":
                            analysis_type_param = "structural_prediction"
                        
                        results = analyze_sequence_with_ai(
                            sequence=cleaned_sequence,
                            analysis_type=analysis_type_param
                        )
                        
                        if results:
                            st.session_state.ai_analysis_results = results
                            st.success("AI analysis complete!")
                            
                            if "error" in results:
                                st.error(results["error"])
                                if "raw_result" in results:
                                    with st.expander("Raw AI Response"):
                                        st.text(results["raw_result"])
                            else:
                                # Display the results based on the analysis type
                                if analysis_type_param == "function_prediction" and "predicted_functions" in results:
                                    st.subheader("Predicted Functions")
                                    
                                    for i, func in enumerate(results["predicted_functions"]):
                                        confidence = results.get("confidence", [])[i] if i < len(results.get("confidence", [])) else "Unknown"
                                        rationale = results.get("rationale", [])[i] if i < len(results.get("rationale", [])) else ""
                                        
                                        with st.expander(f"Function {i+1}: {func}"):
                                            st.write(f"**Confidence:** {confidence}")
                                            st.write(f"**Rationale:** {rationale}")
                                
                                elif analysis_type_param == "structural_prediction":
                                    st.subheader("Structural Predictions")
                                    
                                    # Secondary structure
                                    if "secondary_structure" in results:
                                        st.write("**Secondary Structure Prediction:**")
                                        st.write(results["secondary_structure"])
                                    
                                    # Domains
                                    if "domains" in results:
                                        st.write("**Potential Domains:**")
                                        for domain in results["domains"]:
                                            st.write(f"• {domain}")
                                    
                                    # Motifs
                                    if "motifs" in results:
                                        st.write("**Structural Motifs:**")
                                        for motif in results["motifs"]:
                                            st.write(f"• {motif}")
                                    
                                    # Stability
                                    if "stability" in results:
                                        st.write("**Stability Prediction:**")
                                        st.write(results["stability"])
                        else:
                            st.error("Could not analyze the sequence. Please check your API key and try again.")
                    except Exception as e:
                        st.error(f"Error in AI analysis: {str(e)}")
    
    # Protein Structure Prediction Tab
    with ai_tab[3]:
        st.subheader("Protein Structure Prediction")
        st.write("Use language models to predict protein structure properties.")
        
        # Input sequence
        protein_sequence = st.text_area(
            "Enter Protein Sequence",
            height=200,
            placeholder="Paste a protein sequence here...",
            help="Input raw amino acid sequence without headers or formatting"
        )
        
        # Model selection
        model_id = st.selectbox(
            "Protein Language Model",
            ["facebook/esm2_t33_650M_UR50D", "facebook/esm2_t12_35M_UR50D"],
            help="Select a protein language model to use for prediction"
        )
        
        predict_structure_button = st.button("Predict Structure", use_container_width=True)
        
        if predict_structure_button and protein_sequence:
            # Clean and validate the sequence
            cleaned_sequence = ''.join(c for c in protein_sequence if c.isalpha())
            valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
            
            if not all(aa in valid_aa for aa in cleaned_sequence.upper()):
                st.warning("The sequence contains characters not found in standard protein sequences (ACDEFGHIKLMNPQRSTVWY).")
            
            with st.spinner("Running protein structure prediction..."):
                try:
                    result = huggingface_protein_prediction(
                        sequence=cleaned_sequence,
                        model_id=model_id
                    )
                    
                    if result:
                        st.success("Prediction complete!")
                        
                        # Display embedding information
                        st.subheader("Protein Embedding")
                        st.write("The protein language model has converted your protein sequence into a mathematical representation (embedding).")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Sequence Length", result["sequence_length"])
                            st.metric("Unique Amino Acids", result["unique_amino_acids"])
                        
                        with col2:
                            st.metric("Most Common Amino Acid", result["most_common_aa"])
                            st.metric("Model", result["model_id"].split("/")[-1])
                        
                        st.info("""
                        **What does this mean?**
                        
                        Protein language models like ESM (Evolutionary Scale Modeling) learn the patterns and relationships
                        in protein sequences from millions of natural proteins. These embeddings capture information about
                        the protein's structure, function, and evolutionary relationships. They can be used for various tasks
                        such as function prediction, structure prediction, and protein design.
                        """)
                    else:
                        st.error("Could not generate predictions. Please try again with a valid protein sequence.")
                except Exception as e:
                    st.error(f"Error in protein structure prediction: {str(e)}")
