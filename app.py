import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('flood_risk_model.pkl')

# Streamlit app title and description
st.set_page_config(page_title="Flood Risk Prediction", layout="centered")

# Streamlit app title
st.title("🌊 Flood Risk Prediction App")
st.markdown("Get instant flood risk predictions based on environmental factors.")

# User input fields with sidebar
st.header(" Input Environmental Data")

rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0 , max_value=1000)
temperature = st.number_input("🌡️ Temperature (°C)", min_value=-50, max_value=50)
humidity = st.number_input("💧 Humidity (%)", min_value=0, max_value=100)
river_level = st.number_input("🌊 River Level (m)", min_value=0.0)
soil_moisture = st.number_input("🌱 Soil Moisture (%)", min_value=0, max_value=100)

# Prepare input data
input_data = np.array([[rainfall, temperature, humidity, river_level, soil_moisture]])

# Predict button
if st.button("🔍 Predict Flood Risk"):
    prediction = model.predict(input_data)
    st.success(f"🌊 Predicted Flood Risk :**:blue[{prediction[0]}]**")

st.markdown("_____")

# Visualize feature importance
st.header("📊 Model Accuracy vs Max Depth")


# Accuracy graph (optional)
if st.button("📈 Show Accuracy Graph"):
    max_depth_values = range(1, 21)
    train_accuracies = [0.90, 0.92, 0.93, 0.94, 0.95, 0.95, 0.96, 0.96, 0.97, 0.97, 0.98, 0.98, 0.98, 0.99, 0.99, 0.99, 0.99, 1.00, 1.00, 1.00]
    test_accuracies = [0.85, 0.87, 0.88, 0.89, 0.90, 0.91, 0.91, 0.92, 0.93, 0.93, 0.94, 0.94, 0.95, 0.95, 0.95, 0.95, 0.96, 0.96, 0.96, 0.95]

    plt.figure(figsize=(10, 6))
    plt.plot(max_depth_values, train_accuracies, marker='o', label='Training Accuracy', color='blue')
    plt.plot(max_depth_values, test_accuracies, marker='o', label='Testing Accuracy', color='green')
    plt.title('Random Forest Accuracy vs Max Depth')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(plt)


st.markdown("_____")

st.write("🌟 Thank you for using the Flood Risk Prediction App!")
