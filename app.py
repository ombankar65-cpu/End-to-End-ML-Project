import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Student Performance Prediction App")
st.write("Enter the student details below to predict the classification.")

# Layout with two columns
col1, col2 = st.columns(2)

with col1:
    gender_raw = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=18)
    study_hours = st.number_input("Study Hours per Week", min_value=0.0, value=10.0)
    attendance = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=90.0)

with col2:
    parent_edu_raw = st.selectbox("Parent Education Level", 
                                  options=["None", "High School", "Bachelor's", "Master's", "PhD"])
    internet_raw = st.selectbox("Internet Access", options=["Yes", "No"])
    extra_raw = st.selectbox("Extracurricular Activities", options=["Yes", "No"])
    prev_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0, value=75.0)
    final_score = st.number_input("Current Final Score", min_value=0.0, max_value=100.0, value=80.0)

# 2. Mapping Dictionaries 
# (Note: These MUST match the numbers you used during your model training)
gender_map = {"Male": 1, "Female": 0}
edu_map = {"None": 0, "High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
binary_map = {"Yes": 1, "No": 0}

# 3. Create the input DataFrame using the mapped numeric values
input_data = pd.DataFrame({
    'gender': [gender_map[gender_raw]],
    'age': [age],
    'study_hours_per_week': [study_hours],
    'attendance_rate': [attendance],
    'parent_education': [edu_map[parent_edu_raw]],
    'internet_access': [binary_map[internet_raw]],
    'extracurricular': [binary_map[extra_raw]],
    'previous_score': [prev_score],
    'final_score': [final_score]
})

# 4. Prediction Button
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        st.subheader("Result:")
        st.success(f"The predicted classification is: {prediction[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
