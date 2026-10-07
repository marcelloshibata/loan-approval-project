#%%
import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5050/")
models = mlflow.search_registered_models(filter_string="name = 'model_pr'")
latest_version = max([i.version for i in models[0].latest_versions])

model = mlflow.sklearn.load_model(f"models:/model_pr/{latest_version}")
features = model.feature_names_in_

# %%
import streamlit as st
st.title("Loan Approval Project")

age = st.number_input(label="Insert your age", format="%0.1f")
gender = st.radio("Your Gender", options=['Male', 'Female'])
education = st.selectbox(label="Education level", options=['Associate', 'High School','Bachelor', 'Doctorate', 'Master'])
income = st.number_input(label="Your Income", format="%0.1f")
emp_exp = st.slider(label="Years of Employment Experience", max_value=100)
home_own = st.selectbox(label="Home Ownership", options=['Own', 'Rent', 'Mortgage', 'Other'])
loan_amnt = st.number_input(label="Loan Amount")
loan_intent = st.selectbox(label="Loan Intent", options=[
    'Personal',
    'Debt Consolidation',
    'Medical',
    'Venture',
    'Home Improvement',
    'Education'
])
loan_int_rate = st.number_input(label="Loan Interest Rate", format="%0.2f")
loan_percent_income = st.slider(label="Loan amount as a percentage of annual income", max_value=1.00, min_value=0.00)
cb_person = st.number_input(label='Length of credit history in years', max_value=500)
credit_score = st.number_input(label='Credit Score', min_value=0, max_value=1000)
prev = st.radio(label='Indicator of previous loan defaults', options=['Yes', 'No'])

data = {
    'person_age': age,
    'person_gender': gender,
    'person_education': education,
    'person_income': income,
    'person_emp_exp': emp_exp,
    'person_home_ownership': home_own.replace(" ", "").upper(),
    'loan_amnt': loan_amnt,
    'loan_intent': loan_intent,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'cb_person_cred_hist_length': cb_person,
    'credit_score': credit_score,
    'previous_loan_defaults_on_file': prev
}