import streamlit as st
import pickle
import numpy as np

# Load the trained model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please ensure it is in the same directory.")

def main():
    st.title("Student Success Prediction App")
    st.write("Enter the following details to predict the outcome:")

    # Creating input fields for the 9 features
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=100, value=20)
    study_hours = st.number_input("Study Hours per Week", min_value=0.0, value=10.0)
    attendance = st.slider("Attendance Rate (%)", 0, 100, 85)
    parent_edu = st.selectbox("Parent Education Level", ["Low", "Medium", "High"])
    internet = st.radio("Internet Access", ["Yes", "No"])
    extracurricular = st.radio("Extracurricular Activities", ["Yes", "No"])
    prev_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0, value=75.0)
    final_score = st.number_input("Final Score", min_value=0.0, max_value=100.0, value=80.0)

    # Convert categorical inputs to numeric if your model expects numbers
    # (Update these mappings based on how your model was trained)
    gender_val = 1 if gender == "Male" else 0
    internet_val = 1 if internet == "Yes" else 0
    extra_val = 1 if extracurricular == "Yes" else 0
    
    # Simple mapping for education
    edu_map = {"Low": 0, "Medium": 1, "High": 2}
    parent_edu_val = edu_map[parent_edu]

    # Prepare the input array
    features = np.array([[gender_val, age, study_hours, attendance, 
                          parent_edu_val, internet_val, extra_val, 
                          prev_score, final_score]])

    if st.button("Predict"):
        prediction = model.predict(features)
        result = prediction[0]
        
        if result == "Yes":
            st.success(f"Prediction: {result}")
        else:
            st.warning(f"Prediction: {result}")

if __name__ == '__main__':
    main()
