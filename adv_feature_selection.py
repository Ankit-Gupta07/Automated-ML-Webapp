import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

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

    st.markdown("<div class='header'>Advanced Feature Selection</div>", unsafe_allow_html=True)

    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        
        # Dropdown for selecting the label column (target column)
        label_column = st.selectbox("Select the label column (target)", df.columns)

        #Advanced feature selection
        # Assuming 'df' is your DataFrame and 'label_column' is your target variable
        st.write("---")
        st.subheader("1. Advanced feature selection:-")
        st.write(df)
    
        df = df.apply(pd.to_numeric, errors='coerce')  # Converts all columns to numeric
        X = df.drop(columns=[label_column])  # Features
        y = df[label_column]  # Target
        X = df.drop(columns=[label_column])
        y = df[label_column]

        #st.write(X)
        #st.write(y)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #st.write(X_train)

        if X_train.isnull().sum().sum() > 0 or y_train.isnull().sum() > 0:
            st.error("Missing values detected in training data! Please re-check preprocessing.")
        else:
            st.success("No missing values in training data. Proceeding with model training.")
        
        # --- Ridge Regression ---
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        # Important features based on Ridge
        ridge_importance = np.abs(ridge.coef_)
        ridge_important_features = X.columns[ridge_importance > 0.05]  # Adjust threshold as needed


        # --- Lasso Regression ---
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_train, y_train)
        # Important features based on Lasso
        lasso_important_features = X.columns[lasso.coef_ != 0]  # Non-zero coefficients indicate importance

        # --- ElasticNet Regression ---
        elasticnet = ElasticNet(alpha=0.01, l1_ratio=0.5)  # 0.5 balance between Lasso and Ridge
        elasticnet.fit(X_train, y_train)
        # Important features based on ElasticNet
        elasticnet_important_features = X.columns[elasticnet.coef_ != 0]

        # --- Random Forest Feature Importance ---
        # Check the type of label column
        if y_train.dtype in ['object', 'category', 'bool']:  # Categorical target
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # Numerical target
            rf = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the model
        rf.fit(X_train, y_train)

        # Feature importance from Random Forest
        rf_importance = rf.feature_importances_

        # Important features based on threshold
        rf_important_features = X.columns[rf_importance > 0.05]  # Adjust threshold as needed

        # --- Gradient Boosting --- 
        # Check label_column data type
        if y_train.dtype in ['category', 'object', 'bool']:  # For categorical target
            # Gradient Boosting Classifier
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            gb.fit(X_train, y_train)

            # Feature importances
            gb_importance = gb.feature_importances_

            # Important features based on threshold
            gb_important_features = X.columns[gb_importance > 0.05]  # Adjust threshold as needed

        else:  # For continuous (numeric) target
            # Gradient Boosting Regressor
            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb.fit(X_train, y_train)

            # Feature importances
            gb_importance = gb.feature_importances_

            # Important features based on threshold
            gb_important_features = X.columns[gb_importance > 0.05]  # Adjust threshold as needed


        # --- Combined Insights Table (UPDATED) ---
        st.markdown("#### **Coefficents of cols:-**")

        # Create DataFrame combining coefficients and importance
        combined_df = pd.DataFrame({
            'Feature': X.columns,
            'Ridge': ridge.coef_,
            'Lasso': lasso.coef_,
            'ElasticNet': elasticnet.coef_,
            'Random Forest': rf_importance,
            'Gradient Boosting': gb_importance
        })

        # Classify importance levels
        def classify_importance(row):
            if (abs(row['Ridge']) > 0.07 or abs(row['Lasso']) > 0.07 or 
                abs(row['ElasticNet']) > 0.07 or row['Random Forest'] > 0.07 or 
                row['Gradient Boosting'] > 0.07):
                return "Highly Important"
            elif (abs(row['Ridge']) > 0.04 or abs(row['Lasso']) > 0.04 or 
                abs(row['ElasticNet']) > 0.04 or row['Random Forest'] > 0.05 or 
                row['Gradient Boosting'] > 0.05):
                return "Moderately Important"
            elif (abs(row['Ridge']) > 0.02 or abs(row['Lasso']) > 0.02 or 
                abs(row['ElasticNet']) > 0.02 or row['Random Forest'] > 0.02 or 
                row['Gradient Boosting'] > 0.02):
                return "Important"
            else:
                return "Less Important"

        combined_df['Final Importance'] = combined_df.apply(classify_importance, axis=1)

        # Display combined table
        st.dataframe(combined_df)

        # --- Insights and Recommendations ---
        st.write("---")
        st.subheader("**2. Final important features:-**")

        # Display Highly Important Features
        highly_important = combined_df[combined_df['Final Importance'] == 'Highly Important']
        st.write("**Highly Important:**")
        st.write(highly_important[['Feature', 'Ridge', 'Lasso', 'ElasticNet', 'Random Forest', 'Gradient Boosting']])

        # Display Moderately Important Features
        moderately_important = combined_df[combined_df['Final Importance'] == 'Moderately Important']
        st.write("**Moderately Important:**")
        st.write(moderately_important[['Feature', 'Ridge', 'Lasso', 'ElasticNet', 'Random Forest', 'Gradient Boosting']])

        # Display Important Features
        important = combined_df[combined_df['Final Importance'] == 'Important']
        st.write("**Important:**")
        st.write(important[['Feature', 'Ridge', 'Lasso', 'ElasticNet', 'Random Forest', 'Gradient Boosting']])

        # --- Recommendations ---
        st.subheader("**Recommendations for feature selection:-**")

        st.markdown("""
        **Prioritize Highly Important Features:**
        - Focus heavily on features consistently scoring high, such as **Tenure, Complain, and CashbackAmount**.

        **Moderate Importance for Refinement:**
        - Refine features like **SatisfactionScore, DaySinceLastOrder, and MaritalStatus_Single** based on moderate impact.

        **Drop Insignificant Features:**
        - Remove features with near-zero coefficients in all models, e.g., **PreferredLoginDevice, Gender_Male, and PreferedOrderCat_Grocery**.
        """)

        # --- Optional PCA Step ---
        # --- Optional PCA Step ---
        st.write("---")
        st.subheader("**3. Optional PCA for Dimensionality Reduction**")

        # Checkbox to enable PCA
        apply_pca = st.checkbox("Apply PCA for dimensionality reduction?")

        if apply_pca:
            st.write("PCA will reduce dimensionality while retaining most variance.")

            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # Standardize the data before PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Perform PCA
            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)

            # Display explained variance ratio
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)

            # Plot explained variance
            st.subheader("**Explained Variance by Principal Components:**")
            fig, ax = plt.subplots()
            ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('Cumulative Explained Variance')
            ax.grid(True)
            st.pyplot(fig)

            # Allow user to select the number of components
            n_components = st.slider(
                "Select the number of components to retain:",
                min_value=1, max_value=len(X.columns), value=len(X.columns) // 2
            )

            # Apply PCA with selected components
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)

            st.write(f"PCA reduced features from {X.shape[1]} to {n_components}.")

            # --- Final List of Features for Prediction ---
            st.subheader("**Final Features for Prediction:**")

            st.write("PCA-transformed components will be used as features for prediction.")
            final_features = [f"PC{i+1}" for i in range(n_components)]

            st.write("**Final Features:**", final_features)

            pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
            st.write("PCA Transformed Data:")
            st.dataframe(pca_df)
        else:
            # If PCA is not selected, skip PCA process
            st.write("PCA is not applied. Selected features based on importance will can be used for prediction.")
            final_features = combined_df[combined_df['Final Importance'] != 'Less Important']['Feature'].tolist()
            #st.write("**Final Features:**", final_features)

        # --- Manual Feature Selection ---
        st.write("---")
        st.subheader("**4. Manual Feature Selection**")

        st.write("""
        Based on the analysis above and domain knowledge, you can manually select the final features to be used for training and prediction.
        """)

        # Combine all features (original and PCA-transformed if applied)
        if apply_pca:
            all_features = [f"PC{i+1}" for i in range(n_components)]
        else:
            all_features = X.columns.tolist()

        # Create multiselect box for manual selection
        final_selected_features = st.multiselect(
            "Select the features you want to include in the final model:",
            options=all_features,
            default=all_features  # Pre-select all features initially
        )

        st.write("**Final Selected Features for Prediction:**", final_selected_features)

        # Save final features for later use
        st.session_state['final_features'] = final_selected_features

        st.write("We are done with feature selection, let's now move to building the model!")





        
        
        

    
    
    
    
    else:
        st.info("No dataset selected.")
