import streamlit as st
import numpy as np
import joblib

# Load saved machine learning model and feature scaler
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("Heart Disease Risk Prediction")
st.write("Enter patient clinical features below to evaluate heart disease risk.")

# Input fields organized in two columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
    cp = st.selectbox("Chest Pain Type", options=[("Typical Angina", 1), ("Atypical Angina", 2), ("Non-anginal", 3), ("Asymptomatic", 4)], format_func=lambda x: x[0])[1]
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120)
    chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

with col2:
    restecg = st.selectbox("Resting ECG Results", options=[("Normal", 0), ("ST-T Wave Abnormality", 1), ("Left Ventricular Hypertrophy", 2)], format_func=lambda x: x[0])[1]
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST", options=[("Upsloping", 1), ("Flat", 2), ("Downsloping", 3)], format_func=lambda x: x[0])[1]
    ca = st.selectbox("Major Vessels Colored by Flourosopy", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[("Normal", 3), ("Fixed Defect", 6), ("Reversable Defect", 7)], format_func=lambda x: x[0])[1]

st.markdown("---")

# Execute prediction when button is clicked
if st.button("Predict Patient Risk", type="primary"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.error("⚠️ Prediction Result: High Risk of Heart Disease")
    else:
        st.success("✅ Prediction Result: Low Risk of Heart Disease")