import streamlit as st
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample

# Add CSS Styling
def add_funky_css():
    st.markdown("""
        <style>
            body {
                background-color: #f4f4f4;
                font-family: 'Roboto', sans-serif;
                color: #333;
            }
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

# Analyze and Handle Imbalanced Data
def analyze_and_handle_imbalance():
    st.markdown("<div class='header'>Handle Imbalanced Data</div>", unsafe_allow_html=True)

    # Check if dataset is available
    if 'df' not in st.session_state or st.session_state['df'] is None:
        st.info("No dataset available. Please upload a dataset first.")
        return

    df = st.session_state['df']
    st.write(df.head())

    # Step 1: Ask for Label Column
    label_column = st.selectbox("Select the label column (target)", df.columns)

    # Step 2: Ask for Task Type
    task_type = st.radio("Select the task type", ["Classification", "Regression"])

    # Step 3: Check Imbalance in Label Column
    st.subheader("Label Column Imbalance Analysis")
    label_counts = df[label_column].value_counts()
    st.bar_chart(label_counts)

    imbalance_ratio = label_counts.max() / label_counts.min()
    if imbalance_ratio > 1.5:
        st.warning("The label column appears imbalanced.")
        st.write("Suggested Methods:")
        if task_type == "Classification":
            st.write("- Oversampling (SMOTE)")
            st.write("- Undersampling")
        elif task_type == "Regression":
            st.write("- SMOTE for Regression")
    else:
        st.success("The label column appears balanced.")

    # Step 4: Handle Label Imbalance (if needed)
    if imbalance_ratio > 1.5:
        handle_label = st.checkbox("Apply Balancing to Label Column")
        if handle_label:
            if task_type == "Classification":
                strategy = st.radio("Select balancing method", ["Oversampling (SMOTE)", "Undersampling"])
                if strategy == "Oversampling (SMOTE)":
                    smote = SMOTE(random_state=42)
                    X = df.drop(columns=[label_column])
                    y = df[label_column]
                    X_resampled, y_resampled = smote.fit_resample(X, y)
                else:
                    rus = RandomUnderSampler(random_state=42)
                    X = df.drop(columns=[label_column])
                    y = df[label_column]
                    X_resampled, y_resampled = rus.fit_resample(X, y)

            elif task_type == "Regression":
                smote = SMOTE(random_state=42, k_neighbors=3)
                X = df.drop(columns=[label_column])
                y = df[label_column]
                X_resampled, y_resampled = smote.fit_resample(X, y)

            # Update balanced dataset
            df = pd.DataFrame(X_resampled, columns=X.columns)
            df[label_column] = y_resampled
            st.session_state['df'] = df
            st.success("Label column imbalance handled successfully!")

    # Step 5: Check and Handle Imbalance in Encoded Features
    st.subheader("Categorical Feature Balance Analysis")
    encoded_categories = {}
    for col in df.columns:
        if '_' in col:  # Detect encoded columns by naming pattern
            base_name = col.rsplit('_', 1)[0]
            if base_name not in encoded_categories:
                encoded_categories[base_name] = []
            encoded_categories[base_name].append(col)

    for category, columns in encoded_categories.items():
        st.write(f"### Analyzing: **{category}**")
        counts = {col: df[col].sum() for col in columns}
        counts_df = pd.DataFrame(list(counts.items()), columns=['Category', 'Count'])
        st.dataframe(counts_df)

        max_count = max(counts.values())
        min_count = min(counts.values())
        imbalance_ratio = max_count / (min_count + 1e-5)

        if imbalance_ratio > 1.5:
            st.warning(
                f"**{category}** shows imbalance. Suggestions: \n"
                "- Use Stratified Sampling to preserve ratios. \n"
                "- Apply Balancing Techniques (Oversampling or Undersampling) if needed."
            )
        else:
            st.success(f"**{category}** appears balanced. No action needed.")

    if not encoded_categories:
        st.info("No encoded categorical columns detected.")

add_funky_css()
analyze_and_handle_imbalance()