import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
from scipy.stats import f_oneway
import plotly.express as px
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.preprocessing import LabelEncoder
import plotly.figure_factory as ff
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer

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

    st.markdown("<div class='header'>Feature Selection</div>", unsafe_allow_html=True)

    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        
        # Dropdown for selecting the label column (target column)
        label_column = st.selectbox("Select the label column (target)", df.columns)

        st.subheader("1. Bivariate analysis:-")

        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Analyze each column against the target
        for col in df.columns:
            if col == label_column:  # Skip label column
                continue
            
            with st.expander(f"Column: {col} vs {label_column}"):

                # Deduce Data Types
                col_type = 'numerical' if col in numerical_cols else 'categorical'
                label_type = 'numerical' if label_column in numerical_cols else 'categorical'

                st.write(f"**Feature Column ({col}) Type:** {df[col].dtype}")
                st.write(f"**Label Column ({label_column}) Type:** {df[label_column].dtype}")

                # Case 1: Numerical vs Numerical - Scatter Plot
                if col_type == 'numerical' and label_type == 'numerical':
                    st.markdown(f"### Scatter Plot: **{col} vs {label_column}**")
                    fig = px.scatter(
                        df, x=col, y=label_column, 
                        title=f"{col} vs {label_column}",
                        template="plotly_white",
                        color_discrete_sequence=['skyblue']
                    )
                    fig.update_layout(title_x=0.5)
                    st.plotly_chart(fig, use_container_width=True)

                # Case 2: Numerical vs Categorical - Box Plot
                elif col_type == 'numerical' and label_type == 'categorical':
                    st.markdown(f"### Box Plot1: **{col} vs {label_column}**")
                    fig = px.box(
                        df, x=label_column, y=col,
                        title=f"{col} vs {label_column}",
                        template="plotly_white",
                        color_discrete_sequence=['skyblue']
                    )
                    fig.update_layout(title_x=0.5)
                    st.plotly_chart(fig, use_container_width=True)

                # Case 3: Categorical vs Numerical - Box Plot
                elif col_type == 'categorical' and label_type == 'numerical':
                    st.markdown(f"### Box Plot2: **{col} vs {label_column}**")
                    fig = px.box(
                        df, x=col, y=label_column,
                        title=f"{col} vs {label_column}",
                        template="plotly_white",
                        color_discrete_sequence=['skyblue']
                    )
                    fig.update_layout(title_x=0.5)
                    st.plotly_chart(fig, use_container_width=True)

                # Case 4: Categorical vs Categorical - Stacked Bar Plot
                elif col_type == 'categorical' and label_type == 'categorical':
                    st.markdown(f"### Stacked Bar Plot: **{col} vs {label_column}**")
                    count_data = df.groupby([col, label_column]).size().unstack().fillna(0)
                    fig = px.bar(
                        count_data, 
                        barmode='stack',
                        title=f"{col} vs {label_column}",
                        template="plotly_white"
                    )
                    fig.update_layout(title_x=0.5, xaxis_title=col, yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)

        # If no columns are available
        if not numerical_cols and not categorical_cols:
            st.info("No columns available for bivariate analysis.")

        # Correlation Matrices
        st.write("---")
        st.subheader("2. Correlation matrices:-")

        # Numerical Correlation Matrix
        st.markdown('<p style="color:red; font-size:18px;"><b>a) Numerical Correlation Matrix</b></p>', unsafe_allow_html=True)
        st.markdown("**Why?** Correlation helps identify linear relationships between numerical variables, which can indicate multicollinearity or dependencies.")
        num_df = df.select_dtypes(include=['float64', 'int64'])
        corr_matrix = num_df.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        st.markdown("**Acceptable Range Details:**")
        st.write("- Strong: > 0.7 or < -0.7")
        st.write("- Moderate: 0.3 to 0.7 or -0.3 to -0.7")
        st.write("- Weak: < 0.3 or > -0.3")

        # Categorical Correlation Matrix (Cramér’s V)
        st.write("---")
        st.markdown('<p style="color:red; font-size:18px;"><b>b) Categorical Correlation Matrix (Cramér’s V)</b></p>', unsafe_allow_html=True)
        st.markdown("**Why?** Measures the strength of association between categorical variables, identifying relationships that can impact predictions.")
        cat_df = df.select_dtypes(include=['object', 'category'])

        def cramers_v(x, y):
            confusion_matrix = pd.crosstab(x, y)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

        cramers_results = pd.DataFrame(index=cat_df.columns, columns=cat_df.columns)
        for col1 in cat_df.columns:
            for col2 in cat_df.columns:
                if col1 == col2:
                    cramers_results.loc[col1, col2] = 1.0
                else:
                    cramers_results.loc[col1, col2] = cramers_v(cat_df[col1], cat_df[col2])

        fig = ff.create_annotated_heatmap(
            z=cramers_results.astype(float).round(4).values,
            x=cramers_results.columns.tolist(),
            y=cramers_results.columns.tolist(),
            colorscale='Viridis'
        )
        st.plotly_chart(fig)
        st.markdown("**Acceptable Range Details:**")
        st.write("- Strong Association: > 0.5")
        st.write("- Moderate Association: 0.3 to 0.5")
        st.write("- Weak Association: < 0.3")

        # Statistical Tests
        st.write("---")
        st.subheader("3. Statistical tests:-")

        # Chi-Square Test (Categorical vs Categorical)
        st.markdown('<p style="color:blue; font-size:18px;"><b>a) Chi-Square Test (Categorical vs Categorical)</b></p>', unsafe_allow_html=True)
        st.markdown("**Why?** Determines if two categorical variables are independent or associated.")
        chi_square_results = []
        for col in cat_df.columns:
            if col != label_column:
                contingency_table = pd.crosstab(df[col], df[label_column])
                chi2, p, _, _ = chi2_contingency(contingency_table)
                chi_square_results.append([col, chi2, p])
        chi_square_df = pd.DataFrame(chi_square_results, columns=["Feature", "Chi-Square Value", "p-value"])
        st.write(chi_square_df)
        st.markdown("**Acceptable Range Details:**")
        st.write("- p-value < 0.05: Significant relationship")
        st.write("- p-value >= 0.05: No significant relationship")

        # ANOVA Test (Categorical vs Numerical)
        st.write("---")
        st.markdown('<p style="color:blue; font-size:18px;"><b>b) ANOVA Test (Categorical vs Numerical)</b></p>', unsafe_allow_html=True)
        st.markdown("**Why?** Compares means of numerical values across categorical groups to find significant differences.")
        anova_results = []
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in cat_df.columns:
            if col != label_column:
                groups = [df[df[col] == cat][label_column] for cat in df[col].unique()]
                f_stat, p = f_oneway(*groups)
                anova_results.append([col, f_stat, p])
        anova_df = pd.DataFrame(anova_results, columns=["Feature", "F-Statistic", "p-value"])
        st.write(anova_df)
        st.markdown("**Acceptable Range Details:**")
        st.write("- p-value < 0.05: Significant difference")
        st.write("- p-value >= 0.05: No significant difference")

        # Point-Biserial Correlation (Numerical vs Binary Categorical)
        st.write("---")
        st.markdown('<p style="color:blue; font-size:18px;"><b>c) Point-Biserial Correlation (Numerical vs Binary Categorical)</b></p>', unsafe_allow_html=True)
        st.markdown("**Why?** Measures the correlation between a numerical variable and a binary categorical variable.")

        def is_binary(column):
            return column.nunique() == 2

        binary_results = []
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if col != label_column:
                if is_binary(df[label_column]):
                    corr, p_value = stats.pointbiserialr(df[col].apply(lambda x: 1 if x == df[label_column].iloc[0] else 0), df[label_column])
                    binary_results.append([col, corr, p_value])

        if binary_results:
            binary_df = pd.DataFrame(binary_results, columns=["Feature", "Point-Biserial Correlation", "p-value"])
            st.write(binary_df)
            st.markdown("**Acceptable Range Details:**")
            st.write("- p-value < 0.05: Significant relationship")
            st.write("- p-value >= 0.05: No significant relationship")
        else:
            st.info("No binary categorical features available for Point-Biserial Correlation.")

        # ANOVA/Kruskal-Wallis Test (Numerical vs Categorical)
        st.write("---")
        st.markdown('<p style="color:blue; font-size:18px;"><b>d) ANOVA / Kruskal-Wallis Test (Numerical vs Categorical)</b></p>', unsafe_allow_html=True)
        st.markdown("**Why?** Tests if numerical values differ significantly across groups in a categorical variable.")
        anova_kruskal_results = []
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if col != label_column:
                if df[label_column].nunique() > 2:
                    if is_binary(df[label_column]):
                        groups = [df[df[col] == category][label_column].values for category in df[col].unique()]
                        f_stat, p = stats.kruskal(*groups)
                        anova_kruskal_results.append([col, f_stat, p])
                    else:
                        groups = [df[df[col] == category][label_column].values for category in df[col].unique()]
                        f_stat, p = stats.f_oneway(*groups)
                        anova_kruskal_results.append([col, f_stat, p])

        if anova_kruskal_results:
            anova_kruskal_df = pd.DataFrame(anova_kruskal_results, columns=["Feature", "Statistic", "p-value"])
            st.write(anova_kruskal_df)
            st.markdown("**Acceptable Range Details:**")
            st.write("- p-value < 0.05: Significant difference")
            st.write("- p-value >= 0.05: No significant difference")
        else:
            st.info("No categorical features available for ANOVA / Kruskal-Wallis tests.")

        st.write("---")
        st.subheader("4. Variance Inflation Factor (VIF):-")

        st.markdown("**Why?** VIF quantifies how much the variance of the estimated regression coefficients is inflated due to collinearity with other predictors.")

        # Function to calculate VIF
        def calculate_vif(df):
            # Ensure all columns are numeric
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            df_numeric = df_numeric.fillna(0).astype(float)  # Coerce to numeric and fill NaNs

            # Calculate VIF
            vif_data = pd.DataFrame()
            vif_data["Feature"] = df_numeric.columns
            vif_data["VIF"] = [variance_inflation_factor(df_numeric.values, i) for i in range(df_numeric.shape[1])]
            
            return vif_data

        # Convert categorical columns to numerical (One-Hot Encoding)
        def encode_categorical_columns(df):
            categorical_cols = df.select_dtypes(include=['object']).columns
            encoder = OneHotEncoder(drop='first', sparse_output=False)  # Use sparse_output=False instead of sparse=False
            encoded_data = encoder.fit_transform(df[categorical_cols])
            
            # Create a DataFrame with the encoded columns
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
            
            # Drop original categorical columns and join the encoded ones
            df_encoded = df.drop(columns=categorical_cols).join(encoded_df)
            return df_encoded

        # Select all columns for VIF calculation
        df_encoded = encode_categorical_columns(df)

        # Calculate VIF and display results
        vif_result = calculate_vif(df_encoded)
        st.write("VIF Results:")
        st.write(vif_result)

        # Highlight high VIF values
        st.write("**Features with VIF > 10 may indicate multicollinearity.**")

            # Highlight high VIF values
        st.markdown("**Acceptable Range Details:**")
        st.markdown("- VIF < 5: Low multicollinearity")
        st.markdown("- 5 <= VIF <= 10: Moderate multicollinearity")
        st.markdown("- VIF > 10: High multicollinearity, which can lead to instability in the model")
        
        st.write("---")
        st.subheader("5. Important Columns Based on Analysis:-")
        important_columns = []

        # Numerical Correlation Analysis
        for col in num_df.columns:
            if col != label_column:  # Exclude label column
                correlations = corr_matrix[col].abs()
                high_corr = correlations[correlations > 0.7].index.tolist()
                # Exclude label column and self-correlation
                high_corr = [c for c in high_corr if c != label_column and c != col]
                important_columns.extend(high_corr)

        # Categorical Correlation (Cramér’s V)
        for col in cramers_results.columns:
            if col != label_column:  # Exclude label column
                high_cramer = cramers_results[col][cramers_results[col] > 0.5].index.tolist()
                high_cramer = [c for c in high_cramer if c != label_column]
                important_columns.extend(high_cramer)

        # Chi-Square Test
        chi_significant = chi_square_df[chi_square_df['p-value'] < 0.05]['Feature'].tolist()
        chi_significant = [c for c in chi_significant if c != label_column]  # Exclude label column
        important_columns.extend(chi_significant)

        # ANOVA Test
        anova_significant = anova_df[anova_df['p-value'] < 0.05]['Feature'].tolist()
        anova_significant = [c for c in anova_significant if c != label_column]  # Exclude label column
        important_columns.extend(anova_significant)

        # Point-Biserial Correlation
        if 'binary_df' in locals():  # Only if binary_df exists
            binary_significant = binary_df[binary_df['p-value'] < 0.05]['Feature'].tolist()
            binary_significant = [c for c in binary_significant if c != label_column]  # Exclude label column
            important_columns.extend(binary_significant)

        # ANOVA/Kruskal-Wallis Test
        if 'anova_kruskal_df' in locals():  # Only if anova_kruskal_df exists
            anova_kruskal_significant = anova_kruskal_df[anova_kruskal_df['p-value'] < 0.05]['Feature'].tolist()
            anova_kruskal_significant = [c for c in anova_kruskal_significant if c != label_column]  # Exclude label column
            important_columns.extend(anova_kruskal_significant)

        # VIF Calculation and Identification of High VIF Features
        # Function to calculate VIF
        def calculate_vif(df):
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            df_numeric = df_numeric.fillna(0).astype(float)  # Coerce to numeric and fill NaNs

            # Calculate VIF
            vif_data = pd.DataFrame()
            vif_data["Feature"] = df_numeric.columns
            vif_data["VIF"] = [variance_inflation_factor(df_numeric.values, i) for i in range(df_numeric.shape[1])]
            
            return vif_data

        # Convert categorical columns to numerical (One-Hot Encoding)
        def encode_categorical_columns(df):
            categorical_cols = df.select_dtypes(include=['object']).columns
            encoder = OneHotEncoder(drop='first', sparse_output=False)
            encoded_data = encoder.fit_transform(df[categorical_cols])
            
            # Create a DataFrame with the encoded columns
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
            
            # Drop original categorical columns and join the encoded ones
            df_encoded = df.drop(columns=categorical_cols).join(encoded_df)
            return df_encoded

        # Select all columns for VIF calculation
        df_encoded = encode_categorical_columns(df)

        # Calculate VIF and get high VIF features (VIF > 10)
        vif_result = calculate_vif(df_encoded)
        high_vif_features = vif_result[vif_result["VIF"] > 10]["Feature"].tolist()

        # Add high VIF features to important columns
        important_columns.extend(high_vif_features)

        # Remove duplicates and display
        important_columns = list(set(important_columns))

        # Display the important columns
        if important_columns:
            st.markdown("##### Important columns identified till now:")
            for col in important_columns:
                st.write(f"- {col}")
        else:
            st.info("No important columns identified based on the analysis.")
        
        # Section Header
        st.write("---")
        st.subheader("6. Scaling and Transformation:-")

        # Explanation for Scaling and Transformation
        st.markdown("<p style='color:blue; font-size:18px;'><b>Why?</b></p>", unsafe_allow_html=True)
        st.markdown("Scaling and transformation standardize data, improve model performance, and handle skewed distributions effectively.")

        # --- SCALING ---
        st.markdown("### Scaling:-")
        st.markdown("**Check Ranges of Numerical Columns:**")

        # Display numerical column ranges
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        st.write(df[num_cols].describe().loc[["min", "max", "mean"]])

        # Dropdown for scaling method
        scaler_option = st.selectbox("Select Scaling Method for All Columns:", ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"])

        if st.button("Apply Scaling"):
            if scaler_option != "None":
                # Apply selected scaling
                if scaler_option == "StandardScaler":
                    scaler = StandardScaler()
                elif scaler_option == "MinMaxScaler":
                    scaler = MinMaxScaler()
                elif scaler_option == "RobustScaler":
                    scaler = RobustScaler()

                scaled_data = scaler.fit_transform(df[num_cols])
                df[num_cols] = pd.DataFrame(scaled_data, columns=num_cols)
                st.success(f"Scaling applied using {scaler_option}.")
            else:
                st.warning("No scaling method selected.")

        # --- TRANSFORMATION ---
        st.write("---")
        st.markdown("### Transformation:-")
        st.markdown("**Visualize Distributions of Numerical Columns:**")

        # Show histograms for distributions and detect skewness
        transform_methods = {}
        skewness_info = {}
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

            # Calculate skewness
            skewness = df[col].skew()
            skewness_info[col] = skewness

            # Provide recommendations
            st.write(f"Skewness for {col}: {skewness:.2f}")
            if abs(skewness) > 1:
                st.warning(f"Highly skewed. Consider Power Transformation.")
            elif 0.5 < abs(skewness) <= 1:
                st.info(f"Moderately skewed. Consider Quantile Transformation.")
            else:
                st.success(f"Fairly symmetric. Transformation may not be needed.")

            # Dropdown for transformation method
            transform_methods[col] = st.selectbox(f"Select Transformation Method for {col}", ["None", "PowerTransformer", "QuantileTransformer"], key=col)
            st.write("---")

        if st.button("Apply Transformation"):
            for col, method in transform_methods.items():
                if method != "None":
                    # Apply selected transformation
                    if method == "PowerTransformer":
                        transformer = PowerTransformer(method='yeo-johnson')
                    elif method == "QuantileTransformer":
                        transformer = QuantileTransformer(output_distribution='normal')

                    transformed_data = transformer.fit_transform(df[[col]])
                    df[col] = transformed_data
                    st.success(f"{method} applied to {col}.")

            # Show updated dataframe
            st.write("Updated DataFrame after Scaling and Transformation:")
            st.write(df.head())
        else:
            st.info("Click the buttons above to apply scaling or transformation.")

        # Custom code block - start
        st.write("---")
        st.subheader("7. Custom code:-")
        
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

        # --- ENCODING --- 
        st.write("---")
        st.markdown("### 8. Encoding:-")
        st.markdown("**Select Encoding Method for Categorical Columns:**")

        encoding_methods = {}
        variable_types = {}

        # Identify categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category'])

        # Fill missing values in categorical columns before encoding
        df[cat_cols] = df[cat_cols].fillna('Unknown')

        # Iterate through categorical columns and allow user to select encoding method and variable type
        for col in cat_cols:
            st.write(f"#### {col}")

            # Dropdown for variable type (Ordinal or Nominal)
            variable_types[col] = st.selectbox(
                f"Select Variable Type for {col} (Ordinal or Nominal)",
                ["Nominal", "Ordinal"],
                key=f"type_{col}"
            )

            # Dropdown for encoding method
            encoding_methods[col] = st.selectbox(
                f"Select Encoding Method for {col}",
                ["None", "One-Hot Encoding", "Label Encoding"],
                key=col
            )

            st.write("---")

        # Apply encoding when the button is pressed
        if st.button("Apply Encoding"):
            for col, method in encoding_methods.items():
                if method != "None":
                    # Apply encoding based on variable type
                    if variable_types[col] == "Nominal":
                        if method == "One-Hot Encoding":
                            # One-hot encode the column
                            df = pd.get_dummies(df, columns=[col])
                            st.success(f"One-Hot Encoding applied to {col}.")
                    elif variable_types[col] == "Ordinal":
                        if method == "Label Encoding":
                            # Label encode the column
                            le = LabelEncoder()
                            df[col] = le.fit_transform(df[col])
                            st.success(f"Label Encoding applied to {col}.")

            # Ensure all boolean columns have 0/1 instead of None
            
            st.write("Testing")
            st.write(df)
            df = df.fillna(0).astype(int)

            # Save the updated DataFrame to session state
            st.session_state['df'] = df

            # Diagnostic checks
            st.write("Updated DataFrame after Encoding:")
            st.write(df)
            st.write("Check Data Types:")
            st.write(df.dtypes)
            st.write("Missing Values Check:")
            st.write(df.isnull().sum())
        else:
            st.info("Click the button above to apply encoding.")
             
    
    
    
    else:
        st.info("No dataset selected.")
