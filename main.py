import streamlit as st

# Function to add classy custom CSS
def add_classy_css():
    st.markdown("""
        <style>
            /* Global styling */
            body {
                font-family: 'Roboto', sans-serif;
                color: #ffffff;
            }
            
            /* Sidebar styling */
            .css-1d391kg {
                background: #222222;
                color: white;
                border-radius: 20px;
                padding: 10px;
                font-size: 18px;
                transition: transform 0.3s ease;
            }
            
            .css-1d391kg:hover {
                transform: scale(1.05);
            }

            .css-1lcbmfe {
                background: #ff1e56;
                color: white;
            }

            .stButton>button {
                background-color: #FF8C00;
                color: white;
                border-radius: 30px;
                padding: 15px;
                font-size: 20px;
                width: 100%;
                border: none;
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
                cursor: pointer;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }

            .stButton>button:hover {
                transform: scale(1.1);
                box-shadow: 0 15px 25px rgba(0, 0, 0, 0.5);
            }

            /* Header Styling */
            .header {
                text-align: center;
                font-size: 3.5em;
                color: #2576f9;
                text-transform: uppercase;
                margin-top: 40px;
                font-weight: bold;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
            }

            /* Subtle fade-in effect for the title */
            .title {
                animation: fadeIn 3s ease-in-out;
            }

            @keyframes fadeIn {
                0% {
                    opacity: 0;
                }
                100% {
                    opacity: 1;
                }
            }

            /* Footer Styling */
            .footer {
                text-align: center;
                font-size: 15px;
                color: #F44336;
                margin-top: 40px;
                padding: 10px;
                background-color: #333;
                border-radius: 20px;
            }

            /* Fancy Card Styling */
            .card {
                background-color: #FF5722;
                border-radius: 15px;
                box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin: 20px;
                color: white;
                transition: transform 0.3s ease;
            }

            .card:hover {
                transform: scale(1.05);
            }

            .card h3 {
                text-transform: uppercase;
            }

            .card p {
                font-size: 16px;
            }
        </style>
    """, unsafe_allow_html=True)

# Main app function
def main():
    # Add custom classy CSS
    add_classy_css()

    # Sidebar Navigation with Classy Buttons
    st.sidebar.title("🎉 Navigation Bar 🎉")
    section = st.sidebar.radio(
        "Select an option:",
        ("🏠 Home", "📂 Upload File", "📊 EDA & cleaning - Num cols","📊 EDA & cleaning - Cat cols", "🔧 Feature Selection", "🔧 Advanced Feature Selection", "Handling Imbalanced data" , "🚀 Run ML Models")
    )

    # Home Section
    if section == "🏠 Home":
        st.markdown("<div class='header title'>Welcome to the Automated ML Web App!</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center;'><strong>Explore the full potential of machine learning with ease.</strong></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center;'>Use the sidebar to navigate through different sections.</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><h3>✨ What’s New?</h3><p>This web app lets you upload data, clean it, explore it, and run powerful machine learning models effortlessly.</p></div>", unsafe_allow_html=True)

    # Upload File Section
    elif section == "📂 Upload File":
        import upload_file
        upload_file.display()

    # EDA Section
    elif section == "📊 EDA & cleaning - Num cols":
        import eda_num
        eda_num.display()

    # Data Cleaning Section
    elif section == "📊 EDA & cleaning - Cat cols":
        import eda_cat
        eda_cat.display()   

    # Feature Selection Section
    elif section == "🔧 Feature Selection":
        import feature_selection
        feature_selection.display()

    # Feature Selection Section
    elif section == "🔧 Advanced Feature Selection":
        import adv_feature_selection
        adv_feature_selection.display()

    # Imbalanced data Section
    elif section == "Handling Imbalanced data":
        import imbalanced_data
        imbalanced_data.display()

    # Run ML Models Section
    elif section == "🚀 Run ML Models":
        import ml_models
        ml_models.display()

    # Footer Section
    st.markdown("<div class='footer'>💻 Powered by Classy ML Enthusiast</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
