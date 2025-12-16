import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("heart_disease_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# App title
st.title("Heart Disease Prediction App🫀🩺")
st.write("Enter patient details to predict heart disease")

# User Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex_label = st.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex_label == "Male" else 0
chest_pain = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
bp = st.number_input("Blood Pressure", value=120)
chol = st.number_input("Cholesterol", value=200)
fbs = st.selectbox("FBS over 120", [0, 1])
ekg = st.selectbox("EKG Results", [0, 1, 2])
max_hr = st.number_input("Max Heart Rate", value=150)
angina = st.selectbox("Exercise Angina", [0, 1])
st_depression = st.number_input("ST Depression", value=1.0)
slope = st.selectbox("Slope of ST", [1, 2, 3])
vessels = st.selectbox("Number of Vessels Fluro", [0, 1, 2, 3])
thallium = st.selectbox("Thallium", [3, 6, 7])

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[ 
        age, sex, chest_pain, bp, chol,
        fbs, ekg, max_hr, angina,
        st_depression, slope, vessels, thallium
    ]], columns=[
        "Age", "Sex", "Chest pain type", "BP", "Cholesterol",
        "FBS over 120", "EKG results", "Max HR",
        "Exercise angina", "ST depression",
        "Slope of ST", "Number of vessels fluro", "Thallium"
    ])

    # Scale input
    scaled_data = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.error(" Heart Disease Detected🫀🩺")
    else:
        st.success(" No Heart Disease🫀🩺")
