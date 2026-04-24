import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Student Performance Prediction App")
st.write("Enter the student details below to predict the classification.")

# Creating columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=18)
    study_hours = st.number_input("Study Hours per Week", min_value=0.0, value=10.0)
    attendance = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=90.0)
    parent_edu = st.selectbox("Parent Education Level", options=["High School", "Bachelor's", "Master's", "PhD", "None"])

with col2:
    internet = st.selectbox("Internet Access", options=["Yes", "No"])
    extracurricular = st.selectbox("Extracurricular Activities", options=["Yes", "No"])
    prev_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0, value=75.0)
    final_score = st.number_input("Current Final Score", min_value=0.0, max_value=100.0, value=80.0)

# Map categorical inputs to match how the model was trained
# Note: You may need to adjust these mappings based on your specific Label Encoding or One-Hot Encoding
input_data = pd.DataFrame({
    'gender': [gender],
    'age': [age],
    'study_hours_per_week': [study_hours],
    'attendance_rate': [attendance],
    'parent_education': [parent_edu],
    'internet_access': [internet],
    'extracurricular': [extracurricular],
    'previous_score': [prev_score],
    'final_score': [final_score]
})

if st.button("Predict"):
    prediction = model.predict(input_data)
    
    st.subheader("Result:")
    if prediction[0] == 1:
        st.success("The model predicts a positive outcome/status.")
    else:
        st.info(f"The predicted class is: {prediction[0]}")
