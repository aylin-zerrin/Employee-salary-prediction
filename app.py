import streamlit as st
import pandas as pd
import joblib

model = joblib.load("salary_model.pkl")

st.title("Employee Salary Prediction App")
st.write("This app predicts whether an employee earns >50K or <=50K per year.")

def user_input():
    age = st.number_input("Age", 18, 100, 30)
    workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Self-emp-inc', 'Without-pay'])
    fnlwgt = st.number_input("Final Weight (FNLWGT)", 10000, 1000000, 200000)
    education = st.selectbox("Education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc'])
    edu_num = st.number_input("Education Number", 1, 16, 9)
    marital_status = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced'])
    occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty'])
    relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Unmarried'])
    race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    capital_gain = st.number_input("Capital Gain", 0, 99999, 0)
    capital_loss = st.number_input("Capital Loss", 0, 99999, 0)
    hours = st.number_input("Hours-per-week", 1, 99, 40)
    country = st.selectbox("Native Country", ['United-States', 'India', 'Mexico', 'Philippines'])

    data = {
        'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt, 'education': education,
        'educational-num': edu_num, 'marital-status': marital_status,
        'occupation': occupation, 'relationship': relationship, 'race': race,
        'gender': gender, 'capital-gain': capital_gain, 'capital-loss': capital_loss,
        'hours-per-week': hours, 'native-country': country
    }

    return pd.DataFrame([data])

input_df = user_input()
if st.button("Predict Salary"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Income Category: {prediction}")
