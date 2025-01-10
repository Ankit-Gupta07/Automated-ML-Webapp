import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

        # General Information about the Dataset
        st.markdown("""
            <h3 style="text-align: center; color: #2D3748;">General Information about the Dataset</h3>
            <p style="text-align: center; color: #4A5568; font-size: 16px;">Get a quick overview of the dataset's structure and key statistics.</p>
        """, unsafe_allow_html=True)

        st.write("---")  # Horizontal line for separation

        # Number of rows and columns
        st.subheader("**Dataset shape:-**")
        st.write(f"**Number of rows:** {df.shape[0]}")
        st.write(f"**Number of columns:** {df.shape[1]}")
        st.write("Dataset -")
        st.write(df)

        st.write("---")  # Horizontal line for separation

        # Missing values and data types
        st.subheader("**Checking missing values/data types:-**")
        missing_values = df.isnull().sum()
        data_types = df.dtypes

        # Calculate missing value percentage
        missing_percentage = ((missing_values / len(df)) * 100).round(2)

        # Summary table
        summary_df = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': data_types.values,
            'Missing Values': missing_values.values,
            'Missing Value Percentage (%)': missing_percentage.values
        })

        summary_df.reset_index(drop=True, inplace=True)

        # Display the summary table
        st.dataframe(summary_df, use_container_width=True)

        st.write("---")  # Horizontal line for separation

        # Data type distribution summary
        dtype_summary = df.dtypes.value_counts()
        st.subheader("**Data type summary:-**")
        st.write(dtype_summary)

        st.write("---")  # Horizontal line for separation

        #Numerical cols - EDA start
        st.markdown(
            """
            <div style="text-align: center;">
                <h1><span style="color:blue;">Numerical columns - EDA</span></h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        numeric_columns = df.select_dtypes(include=['number'])
        num_numerical_columns = len(numeric_columns.columns)
        st.markdown(
            f"<h4 style='font-size:20px;'>No. of numerical cols: {num_numerical_columns}</h4>",
            unsafe_allow_html=True
        )

         # Basic statistics for numerical columns
        st.subheader("**1. Basic stats for num cols:-**")
        st.write(df.describe())

        st.write("---")  # Horizontal line for separation

        # Univariate analysis: distribution, boxplot, and KDE for numerical columns
        
        st.subheader("**2. Distribution of Numerical Columns**")

        if len(numeric_columns) > 0:
            for col in numeric_columns:
                with st.expander(f"View Distribution of **{col}**", expanded=False):
                    #st.markdown(f"### **{col}**: Distribution + KDE + Boxplot")

                    # Create a subplot for Distribution+KDE and Boxplot
                    fig = go.Figure()

                    # Histogram + KDE
                    fig.add_trace(go.Histogram(
                        x=df[col],
                        nbinsx=20,
                        name="Histogram",
                        marker_color='skyblue',
                        opacity=0.6,
                        histnorm='density'  # Normalize histogram
                    ))

                    # KDE (Kernel Density Estimation)
                    kde_x = np.linspace(df[col].min(), df[col].max(), 1000)
                    kde_y = np.exp(-0.5 * ((kde_x - df[col].mean()) / df[col].std())**2)
                    fig.add_trace(go.Scatter(
                        x=kde_x,
                        y=kde_y,
                        mode='lines',
                        name="KDE",
                        line=dict(color='orange', width=2)
                    ))

                    # Boxplot
                    fig.add_trace(go.Box(
                        y=df[col],
                        name=f"{col} Boxplot",
                        marker_color='orange'
                    ))

                    # Update layout for aesthetics
                    fig.update_layout(
                        title={
                            'text': f"{col}: Distribution + KDE + Boxplot",
                            'x': 0.5,
                            'xanchor': 'center'
                        },
                        xaxis_title=col,
                        yaxis_title="Density / Frequency",
                        template='plotly_dark',  # Optional dark theme
                        showlegend=True
                    )

                    # Display the plot
                    st.plotly_chart(fig, use_container_width=True)

                    # Distribution Summary Table
                    st.markdown(f"#### Summary for **{col}**")
                    st.write("Below is the basic statistical summary of the column:")
                    summary_table = pd.DataFrame({
                        "Metric": ["Mean", "Median", "Standard Deviation", "Min", "Max"],
                        "Value": [
                            round(df[col].mean(), 2),
                            round(df[col].median(), 2),
                            round(df[col].std(), 2),
                            round(df[col].min(), 2),
                            round(df[col].max(), 2)
                        ]
                    })
                    st.table(summary_table)

                    st.write("---")

            st.markdown("##### These distributions will help us determine:")
            st.write("a. How to handle missing values & outliers.")
            st.write("b. Which columns might be converted to categorical type.")
            st.write("---")
        else:
            st.info("No numerical columns found in the dataset.")
            st.write("---")

        #Handling missing values

        # Select numerical columns
        numeric_columns = df.select_dtypes(include=['number']).columns

        # Initialize a dictionary to store the selected options
        missing_value_methods = {}

        # Display aligned columns and dropdowns
        st.subheader("**3. Handle missing values for numeric cols:-**")
        # Filter for numerical columns with missing values
        numeric_columns_with_na = [col for col in df.select_dtypes(include=['number']).columns if df[col].isnull().sum() > 0]

        if numeric_columns_with_na:
            # Initialize a dictionary to store the selected options
            missing_value_methods = {}

            # Display aligned columns and dropdowns
            for col in numeric_columns_with_na:
                col1, col2 = st.columns([2, 3])

                with col1:
                    st.markdown(f"**{col}**")

                with col2:
                    method = st.selectbox(
                        f"Select action for {col}",
                        options=["Mean Imputation", "Median Imputation", "Mode Imputation","KNN Imputation", "Remove Rows"],
                        key=f"missing_{col}"
                    )
                    missing_value_methods[col] = method

            # Button to apply changes
            if st.button("Apply Missing Value Handling"):
                try:
                    for col, method in missing_value_methods.items():
                        if method == "Mean Imputation":
                            df[col].fillna(df[col].mean(), inplace=True)
                        elif method == "Median Imputation":
                            df[col].fillna(df[col].median(), inplace=True)
                        elif method == "Mode Imputation":
                            df[col].fillna(df[col].mode()[0], inplace=True)
                        elif method == "KNN Imputation":
                            from sklearn.impute import KNNImputer
                            imputer = KNNImputer()
                        elif method == "Remove Rows":
                            df = df[df[col].notnull()]

                    # Update session state with the modified DataFrame
                    st.session_state.df = df
                    st.success("Missing values handled successfully!")

                    # Display updated DataFrame
                    st.subheader("Updated DataFrame")
                    st.dataframe(df)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.info("No numerical columns with missing values detected.")

        st.write("---")
        st.subheader("**4. Handle outliers for num cols:-**")
        # Function to identify columns with outliers using the IQR method
        def identify_outlier_columns(dataframe):
            outlier_columns = []
            for col in dataframe.select_dtypes(include=['number']).columns:
                Q1 = dataframe[col].quantile(0.25)
                Q3 = dataframe[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                if ((dataframe[col] < lower_bound) | (dataframe[col] > upper_bound)).any():
                    outlier_columns.append(col)
            return outlier_columns
        
        # Function to calculate outlier percentage
        def calculate_outlier_percentage(column):
            q1 = column.quantile(0.25)  # First quartile (25th percentile)
            q3 = column.quantile(0.75)  # Third quartile (75th percentile)
            iqr = q3 - q1  # Interquartile range
            lower_bound = q1 - 1.5 * iqr  # Lower bound
            upper_bound = q3 + 1.5 * iqr  # Upper bound
            outliers = column[(column < lower_bound) | (column > upper_bound)]
            return len(outliers) / len(column) * 100  # Percentage of outliers

        # Get columns with outliers
        outlier_columns = identify_outlier_columns(df)

        if outlier_columns:
            # Initialize a dictionary to store the selected outlier handling methods
            outlier_methods = {}

            # Display aligned columns and dropdowns
            for col in outlier_columns:
                col1, col2, col3 = st.columns([2, 2, 3])

                with col1:
                    st.markdown(f"**{col}**")

                with col2:
                    outlier_percentage = round(calculate_outlier_percentage(df[col]), 2)
                    st.markdown(f"{outlier_percentage}% of outliers")

                with col3:
                    method = st.selectbox(
                        f"Select action for {col}",
                        options=[
                            "None", 
                            "Remove Outliers (IQR)", 
                            "Cap Outliers (IQR)", 
                            "Cap Outliers (Standard Deviation)"
                        ],
                        key=f"outlier_{col}"
                    )
                    outlier_methods[col] = method

            # Button to apply changes
            if st.button("Apply Outlier Handling"):
                try:
                    for col, method in outlier_methods.items():
                        if method == "Remove Outliers (IQR)":
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

                        elif method == "Cap Outliers (IQR)":
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            df[col] = np.clip(df[col], lower_bound, upper_bound)

                        elif method == "Cap Outliers (Standard Deviation)":
                            mean = df[col].mean()
                            std_dev = df[col].std()
                            lower_bound = mean - 3 * std_dev
                            upper_bound = mean + 3 * std_dev
                            df[col] = np.clip(df[col], lower_bound, upper_bound)

                    # Update session state with the modified DataFrame
                    st.session_state.df = df
                    st.success("Outliers handled successfully!")

                    # Display updated DataFrame
                    st.subheader("Updated DataFrame")
                    st.dataframe(df)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.info("No numerical columns with outliers detected.")

        # Custom code block - start
        st.write("---")
        st.subheader("5. Data manipulation or Feature engineering")
        
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
        st.subheader("We are done with numerical cols, let's now move to categorical cols!")
    else:
        st.write("Please upload a dataset to proceed with EDA.")

