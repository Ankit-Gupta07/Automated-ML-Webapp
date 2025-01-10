import streamlit as st
import pandas as pd

# Function to add custom funky and classy CSS
def add_funky_css():
    st.markdown("""
        <style>
            /* Global styling */
            body {
                background-color: #f4f4f4; /* Light background */
                font-family: 'Roboto', sans-serif;
                color: #333;
            }

            /* Upload Section Styling */
            .upload-section {
                background-color: #ffffff;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 30px;
                margin: 20px 0;
                text-align: center;
            }

            /* Styling for buttons */
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border-radius: 30px;
                padding: 15px 30px;
                font-size: 18px;
                width: auto;
                border: none;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
                cursor: pointer;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }

            .stButton>button:hover {
                transform: scale(1.1);
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
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

            /* Dataframe Display */
            .dataframe {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-top: 30px;
                overflow-x: auto;
            }

            /* Text and instructions */
            .instructions {
                font-size: 1.2em;
                color: #555;
                text-align: center;
                margin-top: 20px;
            }

            /* Footer */
            .footer {
                text-align: center;
                font-size: 1.2em;
                color: #777;
                margin-top: 40px;
            }
        </style>
    """, unsafe_allow_html=True)

# Function to display the upload page
def display():
    # Add custom CSS to enhance the page design
    add_funky_css()

    # Header for the upload page
    st.markdown("<div class='header'>Upload and Display Dataset</div>", unsafe_allow_html=True)

    # File uploader widget
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

    # If the file is uploaded, handle and display it
    if uploaded_file:
        with st.spinner('Processing your file...'):
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                xls = pd.ExcelFile(uploaded_file)
                sheet_names = xls.sheet_names
                sheet = st.selectbox("Select the sheet", sheet_names)
                df = pd.read_excel(uploaded_file, sheet_name=sheet)

        st.session_state['df'] = df  # Store the uploaded file in session state

        # Display the data in a stylized dataframe
        st.markdown("<div class='dataframe'>", unsafe_allow_html=True)
        st.write(df)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='instructions'>If you can see your uploaded data, then go to the next section to start the EDA process.</div>", unsafe_allow_html=True)
    else:
        # Show instructions when no file is uploaded
        st.markdown("<div class='instructions'>Please upload a file to get started.</div>", unsafe_allow_html=True)
