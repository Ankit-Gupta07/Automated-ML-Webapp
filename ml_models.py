import streamlit as st
import pandas as pd

def display():
    st.header("Run Machine Learning Models")

    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        st.write("Proceed with running ML models and comparing their performance.")
        # Add your ML models code here
    else:
        st.write("Please upload a dataset to proceed with running ML models.")
