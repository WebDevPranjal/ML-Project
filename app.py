import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('best_model_final.pkl')

st.set_page_config(page_title="Relative Humidity Predictor", layout="centered")

# App title
st.title("🌤️ Relative Humidity Prediction App")
st.write("Enter the environmental feature values below to get a predicted humidity level.")

# Input fields for all model features (based on your dataset)
feature_names = [
    'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
    'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)',
    'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'AH',
    'Hour', 'Month', 'Day', 'Weekday'
]

inputs = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    inputs.append(value)

if st.button("Predict Humidity"):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)
    st.success(f"🌡️ Predicted Relative Humidity: **{prediction[0]:.2f}%**")
