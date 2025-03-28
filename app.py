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

# Main title
st.title("BioData Explorer 🧬")
st.subheader("A platform for exploring biological research datasets")

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    
    # Only show navigation options if data is loaded
    if st.session_state.data is not None:
        selected_page = st.radio(
            "Go to:",
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
        st.info("Upload a dataset to enable navigation options")
    
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
