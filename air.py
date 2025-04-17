import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
import xgboost as xgb
import pickle

# Load pretrained models (replace 'model_path' with actual paths)
linear_model = pickle.load(open('D:/streamlit/xgb_gridcv1.pkl', 'rb'))
xgb_model = pickle.load(open('D:/streamlit/xgb_randomcv1.pkl', 'rb'))
lasso_model = pickle.load(open('D:/streamlit/lasso_regression_model.pkl', 'rb'))

# Sidebar and app layout
st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Air Quality Prediction")
st.sidebar.image("D:/streamlit/air_image.jpeg", use_column_width=True)
st.sidebar.markdown("Predict air quality using ML models. Choose your model and input parameters.")

# Title and Header
st.title("🌟 Air Quality Prediction App")
st.markdown(
    """
    Welcome to the **Air Quality Prediction** app. Use this tool to predict the air quality 
    (PM 2.5) based on weather and atmospheric conditions.
    """
)

# Input features explanation
st.markdown(
    """
    ### Input Features
    - **T**: Temperature (°C)
    - **TM**: Maximum temperature (°C)
    - **Tm**: Minimum temperature (°C)
    - **SLP**: Sea level pressure (hPa)
    - **H**: Humidity (%)
    - **VV**: Visibility (km)
    - **V**: Wind speed (km/h)
    - **VM**: Maximum wind speed (km/h)
    """
)

# User Input Section
st.markdown("## Enter Input Parameters")
col1, col2, col3, col4 = st.columns(4)

with col1:
    T = st.number_input("Temperature (T)", min_value=-50.0, max_value=50.0, value=25.0)
with col2:
    TM = st.number_input("Max Temp (TM)", min_value=-50.0, max_value=60.0, value=30.0)
with col3:
    Tm = st.number_input("Min Temp (Tm)", min_value=-50.0, max_value=50.0, value=20.0)
with col4:
    SLP = st.number_input("Sea Level Pressure (SLP)", min_value=900.0, max_value=1100.0, value=1010.0)

col5, col6, col7, col8 = st.columns(4)

with col5:
    H = st.number_input("Humidity (H)", min_value=0.0, max_value=100.0, value=50.0)
with col6:
    VV = st.number_input("Visibility (VV)", min_value=0.0, max_value=50.0, value=10.0)
with col7:
    V = st.number_input("Wind Speed (V)", min_value=0.0, max_value=150.0, value=5.0)
with col8:
    VM = st.number_input("Max Wind Speed (VM)", min_value=0.0, max_value=200.0, value=15.0)

# Model Selection
st.sidebar.markdown("### Select Model")
model_choice = st.sidebar.selectbox(
    "Choose a model for prediction:",
    ["Linear Regression", "XGB Regressor", "Lasso Regressor"]
)

# Prediction
st.markdown("## Model Prediction")

# Prepare input for prediction
input_features = np.array([[T, TM, Tm, SLP, H, VV, V, VM]])

if st.button("Predict"):
    if model_choice == "Linear Regression":
        prediction = linear_model.predict(input_features)
    elif model_choice == "XGB Regressor":
        prediction = xgb_model.predict(input_features)
    elif model_choice == "Lasso Regressor":
        prediction = lasso_model.predict(input_features)

    st.success(f"Predicted PM 2.5 Level: {prediction[0]:.2f} µg/m³")
else:
    st.warning("Click the 'Predict' button to see results!")

# Additional Sections
st.sidebar.markdown("### About the Models")
st.sidebar.info(
    """
    - **Linear Regression**: A simple linear approach to modeling the relationship between input features and the target variable.
    - **XGB Regressor**: A gradient boosting algorithm for high-performance predictions.
    - **Lasso Regressor**: A linear model with L1 regularization to avoid overfitting.
    """
)

st.sidebar.markdown("### Resources")
st.sidebar.markdown("[Learn more about Air Quality](https://www.who.int/air-quality)")

# Footer
st.markdown("---")
st.markdown("© 2024 Air Quality App | Designed with ❤️ using Streamlit")
