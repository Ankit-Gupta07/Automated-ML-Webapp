# Automated Machine Learning Web App

This automated Machine Learning Web App simplifies the process of classification and regression tasks. The app is designed to guide users through essential stages of data preparation, feature selection, and model deployment, providing a seamless and user-friendly interface.

**Note:**  
This project is currently tested and run locally using a virtual machine but will be deployed to the cloud in the near future. **Time-series data is not supported** in this version, but it will be included in the next version of the app.

## Features

### **1. Data Upload and Handling**
- **Flexible Data Upload:** Upload data from multiple sources such as CSV files, SQL databases, and APIs.
- **EDA for Numerical Columns:**
  - Displays the shape, statistical information, and missing value percentage.
  - Includes basic statistics for numerical columns and univariate analysis (distribution, boxplot, KDE).
  - Handles missing values using mean, median, mode, or KNN imputer.
  - Detects and handles outliers using IQR and other techniques.
  - Custom code block for advanced analysis.
  
### **2. Categorical Data Handling**
- **EDA for Categorical Columns:**
  - Provides basic statistics for categorical columns, including value counts and data type distribution.
  - Standardizes categories to ensure uniformity.
  - Visualizes categorical data distribution.
  - Handles missing values for categorical columns.
  - Converts data types for both numerical and categorical columns.

### **3. Feature Selection and Analysis**
- **Bivariate Analysis:**
  - Correlation matrices for both numerical and categorical data (Cramer's V).
  - Statistical tests (Chi-square, ANOVA, Point-Biserial, Kruskal-Wallis) to assess relationships between variables.
  - Calculates Variance Inflation Factor (VIF) to check for multicollinearity.
  - Recommends important features based on statistical analysis.
- **Scaling and Transformation:**
  - Scales numerical data and transforms variables for better model performance.
- **Encoding:** Handles encoding of categorical features for model readiness.

### **4. Advanced Feature Selection**
- **Feature Importance:**
  - Identifies important features using Ridge, Lasso, ElasticNet, Random Forest, and Gradient Boosting.
  - Offers an optional PCA for dimensionality reduction with variance graph visualization.
- **Manual Feature Selection:** Allows users to finalize features based on domain knowledge and analysis results.

### **5. Model Running and Recommendations**
- **Model Identification:** Automatically identifies whether the task is classification or regression.
- **Model Evaluation:**
  - Runs various classification/regression models and generates a combined summary report.
  - Recommends the best fit model based on the evaluation results.

## Installation

To run the app locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/AnkitGupta/automated-ml-web-app.git

2. Navigate to the project folder:
   ```bash
   cd automated-ml-web-app

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the Streamlit app:
   ```bash
   streamlit run main.py

## Requirements
- Python 3.8+
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Scipy
- Matplotlib, Seaborn, Plotly
- Other dependencies are listed in requirements.txt

## Future Plans
- Deployment: The app is currently running locally on a virtual machine, but it will be deployed on the cloud soon for public access.
- Time-Series Support: Time-series data support will be introduced in the next version of the app.

## Contributions and Feedback
Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue on the GitHub repository. I appreciate your feedback and am committed to making this app even better.
