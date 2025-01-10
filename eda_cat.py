import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import numpy as np

def add_funky_css():
    st.markdown("""
        <style>
            /* Global styling */
            body {
                background-color: #f4f4f4; /* Light background */
                font-family: 'Roboto', sans-serif;
                color: #333;
            }


            /* Header Styling */
            .header {
                text-align: center;
                font-size: 2.5em;
                color: #2E7D32;
                text-transform: uppercase;
                margin-bottom: 20px;
                font-weight: bold;
            }

        </style>
    """, unsafe_allow_html=True)

def display():

    add_funky_css()

    st.markdown("<div class='header'>Exploratory Data Analysis (EDA)</div>", unsafe_allow_html=True)

    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        #st.write(df)

        #Categorical cols - EDA start
        st.markdown(
            """
            <div style="text-align: center;">
                <h1><span style="color:red;">Categorical columns - EDA</span></h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Basic statistics for categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category'])
        num_categorical_columns = len(categorical_columns.columns)
        st.markdown(
            f"<h4 style='font-size:20px;'>No. of categorical cols: {num_categorical_columns}</h4>",
            unsafe_allow_html=True
        )
        st.subheader("**1. Basic stats for cat cols:-**")
        st.write(categorical_columns.describe())

        st.write("---")  # Horizontal line for separation

        st.subheader("**2. Value counts for cat cols:-**")
        if len(categorical_columns) > 0:
            # Create two columns for side-by-side display
            cols = st.columns(2)

            for idx, col in enumerate(categorical_columns):
                # Get value counts and reset the index
                value_counts = df[col].value_counts().reset_index()
                value_counts.columns = ['Category', 'Count']

                # Add a heading for the column name and style the dataframe
                styled_df = value_counts.style.set_table_styles(
                    [{'selector': 'thead th', 'props': [('background-color', '#6C5B7B'), ('color', 'white')]},
                    {'selector': 'tbody td', 'props': [('background-color', '#F7F7F7')]},
                    {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#E0E0E0')]},
                    {'selector': 'tbody td', 'props': [('padding', '8px')]},
                    {'selector': 'thead th', 'props': [('padding', '12px')]},
                    {'selector': 'th', 'props': [('text-align', 'center')]},
                    {'selector': 'td', 'props': [('text-align', 'center')]},
                    {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('width', '70%')]}]  # Reduced width to 70%
                )
                
                # Alternate between the two columns (1st column and 2nd column)
                col_idx = idx % 2  # Toggle between 0 and 1 for alternating between columns

                # Display column name and table in the corresponding column
                cols[col_idx].markdown(f"##### **{col}**", unsafe_allow_html=True)
                cols[col_idx].dataframe(styled_df, use_container_width=False)  # Set use_container_width to False for controlled width

                # Add spacing between tables
                if (idx + 1) % 2 == 0 and idx != len(categorical_columns) - 1:
                    st.write("<br>", unsafe_allow_html=True)  # Add a bit of space after the 2nd column
                
            st.write("---")

        else:
            st.info("No categorical columns detected.")

        
        st.subheader("**3. Categorical cols standardization:-**")
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if len(categorical_cols) > 0:
            # Initialize replacement dictionary in session state if not present
            if 'replacement_dict_churn' not in st.session_state:
                st.session_state.replacement_dict_churn = {}

            for col in categorical_cols:
                # Collapsible section for each column
                with st.expander(f"Column: **{col}**", expanded=False):
                    st.markdown(f"**Unique values in {col}:**")
                    unique_values = df[col].unique().tolist()
                    
                    # Display unique values and inputs for replacements side-by-side
                    for value in unique_values:
                        col1, col2 = st.columns([1, 3])  # Adjust column widths for better alignment
                        
                        with col1:
                            st.write(f"Original: '{value}'")
                        
                        with col2:
                            replacement = st.text_input(
                                f"Replace '{value}' with:",
                                key=f"{col}_{value}",
                                placeholder="Enter replacement value"
                            )
                            st.session_state.replacement_dict_churn[(col, value)] = replacement

            # Button to apply replacements
            if st.button("Apply Replacements", key="apply_replacements_churn"):
                try:
                    for (col, old_value), new_value in st.session_state.replacement_dict_churn.items():
                        if new_value:  # Only replace if a new value is provided
                            df[col] = df[col].replace(old_value, new_value)

                    st.success("Replacements applied successfully.")
                    st.write("### Updated Data:")
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error processing replacements: {e}")
        else:
            st.warning("No categorical columns found in the dataset.")
        st.markdown("---")  # Add a horizontal line for separation
        

        
        st.subheader("**4. Categorical cols distribution:-**")
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                with st.expander(f"Column: **{col}**", expanded=False):
                    st.markdown(f"**Distribution of values in `{col}`:**")
                    
                    # Calculate value counts
                    value_counts = df[col].value_counts()
                    percentages = value_counts / len(df) * 100  # Calculate percentages
                    
                    # Create side-by-side visualizations
                    col1, col2 = st.columns([1, 2])  # Adjust width for pie chart and bar plot
                    
                    with col1:
                        #st.markdown("**Pie Chart**")
                        fig_pie = px.pie(
                            names=value_counts.index,
                            values=value_counts.values,
                            title=f"Pie Chart",
                            template="plotly_white"
                        )
                        # Center the title
                        fig_pie.update_layout(
                            title={
                                'text': "Pie chart",
                                'x': 0.5,  # Centers the title
                                'xanchor': 'center'
                            }
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        #st.markdown("**Bar Plot**")
                        fig_bar = px.bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            title=f"Bar Plot",
                            labels={'x': col, 'y': 'Count'},
                            text=value_counts.values,
                            template="plotly_white"
                        )
                        # Center the title
                        fig_bar.update_layout(
                            title={
                                'text': "Bar chart",
                                'x': 0.5,  # Centers the title
                                'xanchor': 'center'
                            }
                        )
                        fig_bar.update_traces(textposition="outside")
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Display the table with counts and percentages
                    st.markdown("**Value Counts and Percentages**")
                    df_distribution = pd.DataFrame({
                        "Value": value_counts.index,
                        "Count": value_counts.values,
                        "Percentage (%)": percentages.values.round(2)
                    })
                    st.dataframe(df_distribution, use_container_width=True)

        else:
            st.warning("No categorical columns found in the dataset.")
        
        st.write("---")
        st.subheader("**5. Handle missing values for categorical cols:-**")

        # Filter for categorical columns with missing values
        categorical_columns_with_na = [col for col in df.select_dtypes(include=['object', 'category']).columns if df[col].isnull().sum() > 0]

        if categorical_columns_with_na:
            # Initialize a dictionary to store the selected options
            missing_value_methods_cat = {}

            # Display aligned columns and dropdowns for categorical columns
            for col in categorical_columns_with_na:
                col1, col2 = st.columns([2, 3])

                with col1:
                    st.markdown(f"**{col}**")

                with col2:
                    method = st.selectbox(
                        f"Select action for {col}",
                        options=["Mode Imputation", "Remove Rows", "Use Other Value"],
                        key=f"missing_cat_{col}"
                    )
                    missing_value_methods_cat[col] = method

            # Button to apply changes
            if st.button("Apply Missing Value Handling"):
                try:
                    for col, method in missing_value_methods_cat.items():
                        if method == "Mode Imputation":
                            df[col].fillna(df[col].mode()[0], inplace=True)
                        elif method == "Remove Rows":
                            df = df[df[col].notnull()]
                        elif method == "Use Other Value":
                            # Ask user to specify a replacement value for missing entries
                            other_value = st.text_input(f"Enter replacement value for missing values in {col}:")
                            df[col].fillna(other_value, inplace=True)

                    # Update session state with the modified DataFrame
                    st.session_state.df = df
                    st.success("Missing values handled successfully!")

                    # Display updated DataFrame
                    st.subheader("Updated DataFrame")
                    st.dataframe(df)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.info("No categorical columns with missing values detected.")

        # Section Header
        st.write("---")
        st.subheader("6. Data type conversion:-")

        # Get all columns and their data types
        all_cols = df.columns.tolist()
        df_dtypes = df.dtypes

        for col in all_cols:
            with st.expander(f"Column: {col}"):
                # Display Current Data Type
                current_dtype = df_dtypes[col]
                st.markdown(f"### Current Data Type: **{current_dtype}**")

                # Display Guidelines
                st.markdown("### Guidelines for Data Type Selection:")
                st.write("- **Integer**: Whole numbers, suitable for counts and numeric categories.")
                st.write("- **Float**: Decimal numbers, suitable for continuous numeric values.")
                st.write("- **String**: Textual data or IDs.")
                st.write("- **Category**: Categorical data with a fixed number of distinct values.")
                st.write("- **Boolean**: Binary data (True/False). Useful for flags or binary categories.")

                # Display Histogram for numerical columns
                if pd.api.types.is_numeric_dtype(df[col]):
                    st.markdown(f"### Distribution of **{col}**")

                    fig = px.histogram(
                        df, x=col, nbins=30, 
                        title=f"Distribution of {col}", 
                        template="plotly_white",
                        color_discrete_sequence=['skyblue']
                    )
                    fig.update_layout(
                        title_x=0.5,  # Center align title
                        xaxis_title=col,
                        yaxis_title="Frequency",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Display Bar Chart for categorical columns
                elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                    st.markdown(f"### Distribution of **{col}**")

                    # Count plot for categorical data
                    count_data = df[col].value_counts()
                    fig = px.bar(
                        count_data, 
                        x=count_data.index, 
                        y=count_data.values, 
                        title=f"Distribution of {col}",
                        labels={col: "Categories", "y": "Frequency"},
                        template="plotly_white",
                        color_discrete_sequence=['skyblue']
                    )
                    fig.update_layout(
                        title_x=0.5,  # Center align title
                        xaxis_title=col,
                        yaxis_title="Frequency",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Display Data Type Conversion Options
                st.markdown("### Convert Data Type")

                # Dropdown for data type conversion
                selected_type = st.selectbox(
                    f"Select new data type for **{col}**:",
                    options=["Integer", "Float", "String", "Category", "Boolean"],
                    key=f"convert_{col}"
                )

                # Apply Conversion Button
                if st.button(f"Convert {col} to {selected_type}", key=f"convert_button_{col}"):
                    try:
                        if selected_type == "Integer":
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype(int)
                        elif selected_type == "Float":
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                        elif selected_type == "String":
                            df[col] = df[col].astype(str)
                        elif selected_type == "Category":
                            df[col] = df[col].astype('category')
                        elif selected_type == "Boolean":
                            df[col] = df[col].astype(bool)
                        st.success(f"Converted {col} to {selected_type} successfully!")
                        st.session_state.df = df  # Update session state
                    except Exception as e:
                        st.error(f"Error converting {col}: {e}")

        # Data type distribution summary
        dtype_summary = df.dtypes.apply(lambda x: str(x)).value_counts()
        st.subheader("**Data type summary:-**")
        st.write(dtype_summary)

        # If no columns are found
        if not all_cols:
            st.info("No columns found in the dataset.")

        # Custom code block - start
        st.write("---")
        st.subheader("7. Data manipulation or Feature engineering:-")

        # Provide a text area for the user to input custom code
        code_input = st.text_area("Write your custom code here:", height=300)

        # Provide an example for users to see the expected format
        st.write("""
        **Example:**
        ```python
        # Example: Add a new column
        df['new_column'] = df['existing_column'] * 2

        # Example: Filter rows
        df = df[df['column_name'] > 100]
        ```
        You can access the dataframe using the variable `df`.
        """)

        # Button to run the custom code
        if st.button("Run Custom Code"):
            try:
                # Execute the user-inputted code
                exec(code_input, {'df': df})

                # Show success message and display the updated dataframe
                st.success("Custom code executed successfully!")
                st.dataframe(df)

                # Update the session state with the modified dataframe
                st.session_state['df'] = df
            except Exception as e:
                st.error(f"An error occurred: {e}")
        # Custom code block - end
        st.write("---")
        




        
                   

        










    else:
        st.write("Please upload a dataset to proceed with EDA.")

