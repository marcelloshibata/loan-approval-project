import pandas as pd
import streamlit as st

if "lang" not in st.session_state:
    st.session_state.lang = "PT"

col_title, col_lang = st.columns([0.75, 0.25])

with col_lang:
    btn_label = "🇺🇸 English" if st.session_state.lang == "PT" else "🇧🇷 Português"
    if st.button(btn_label, use_container_width=True, type="primary"):
        st.session_state.lang = "EN" if st.session_state.lang == "PT" else "PT"
        st.rerun()

content = {
    "PT": {
        "title": "Aprovação de Empréstimo",
        "age": "Insira a sua idade",
        "gender": "Seu Gênero",
        "gender_opts": ["Masculino", "Feminino"],
        "education": "Nível de Escolaridade",
        "education_opts": ["Tecnólogo", "Ensino Médio", "Bacharelado", "Doutorado", "Mestrado"],
        "income": "Sua Renda Anual",
        "emp_exp": "Anos de Experiência Profissional",
        "home_own": "Tipo de Moradia",
        "home_own_opts": ["Própria", "Alugada", "Hipoteca", "Outro"],
        "loan_amnt": "Valor do Empréstimo",
        "loan_intent": "Motivo do Empréstimo",
        "intent_opts": ["Pessoal", "Consolidação de Dívidas", "Médico", "Empreendimento", "Reforma da Casa", "Educação"],
        "loan_int_rate": "Taxa de Juros do Empréstimo (%)",
        "loan_percent_income": "Valor do empréstimo como percentual da renda anual",
        "cb_person": "Tempo de histórico de crédito em anos",
        "credit_score": "Pontuação de Crédito (Credit Score)",
        "prev": "Histórico prévio de inadimplência",
        "prev_opts": ["Sim", "Não"],
        "btn_analyze": "Analisar Empréstimo",
        "success_msg": "Alta probabilidade de aprovação! Probabilidade: {prob:.0f}%",
        "warning_msg": "Média probabilidade de aprovação. Probabilidade: {prob:.0f}%",
        "error_msg": "Baixa probabilidade de aprovação. Probabilidade: {prob:.0f}%",
    },
    "EN": {
        "title": "Loan Approval",
        "age": "Insert your age",
        "gender": "Your Gender",
        "gender_opts": ["Male", "Female"],
        "education": "Education level",
        "education_opts": ["Associate", "High School", "Bachelor", "Doctorate", "Master"],
        "income": "Your Income",
        "emp_exp": "Years of Employment Experience",
        "home_own": "Home Ownership",
        "home_own_opts": ["Own", "Rent", "Mortgage", "Other"],
        "loan_amnt": "Loan Amount",
        "loan_intent": "Loan Intent",
        "intent_opts": ["Personal", "Debt Consolidation", "Medical", "Venture", "Home Improvement", "Education"],
        "loan_int_rate": "Loan Interest Rate",
        "loan_percent_income": "Loan amount as a percentage of annual income",
        "cb_person": "Length of credit history in years",
        "credit_score": "Credit Score",
        "prev": "Indicator of previous loan defaults",
        "prev_opts": ["Yes", "No"],
        "btn_analyze": "Loan Analyze",
        "success_msg": "High probability of approval! Probability: {prob:.0f}%",
        "warning_msg": "Medium probability of approval. Probability: {prob:.0f}%",
        "error_msg": "Low probability of approval. Probability: {prob:.0f}%",
    },
}

t = content[st.session_state.lang]

st.title(t["title"])

model = pd.read_pickle("app/model.pkl")

age = st.number_input(label=t["age"], format="%0.1f")
gender_idx = st.radio(t["gender"], options=t["gender_opts"])
education_idx = st.selectbox(label=t["education"], options=t["education_opts"])
income = st.number_input(label=t["income"], format="%0.1f")
emp_exp = st.slider(label=t["emp_exp"], max_value=100)
home_own_idx = st.selectbox(label=t["home_own"], options=t["home_own_opts"])
loan_amnt = st.number_input(label=t["loan_amnt"])
loan_intent_idx = st.selectbox(label=t["loan_intent"], options=t["intent_opts"])
loan_int_rate = st.number_input(label=t["loan_int_rate"], format="%0.2f")
loan_percent_income = st.slider(label=t["loan_percent_income"], max_value=1.00, min_value=0.00)
cb_person = st.number_input(label=t["cb_person"], max_value=500)
credit_score = st.number_input(label=t["credit_score"], min_value=0, max_value=1000)
prev_idx = st.radio(label=t["prev"], options=t["prev_opts"])

gender_map = {t["gender_opts"][0]: "male", t["gender_opts"][1]: "female"}
education_map = {
    t["education_opts"][0]: "Associate",
    t["education_opts"][1]: "High School",
    t["education_opts"][2]: "Bachelor",
    t["education_opts"][3]: "Doctorate",
    t["education_opts"][4]: "Master",
}
home_own_map = {
    t["home_own_opts"][0]: "OWN",
    t["home_own_opts"][1]: "RENT",
    t["home_own_opts"][2]: "MORTGAGE",
    t["home_own_opts"][3]: "OTHER",
}
loan_intent_map = {
    t["intent_opts"][0]: "PERSONAL",
    t["intent_opts"][1]: "DEBTCONSOLIDATION",
    t["intent_opts"][2]: "MEDICAL",
    t["intent_opts"][3]: "VENTURE",
    t["intent_opts"][4]: "HOMEIMPROVEMENT",
    t["intent_opts"][5]: "EDUCATION",
}
prev_map = {t["prev_opts"][0]: "Yes", t["prev_opts"][1]: "No"}

data = {
    "person_age": age,
    "person_gender": gender_map[gender_idx],
    "person_education": education_map[education_idx],
    "person_income": income,
    "person_emp_exp": emp_exp,
    "person_home_ownership": home_own_map[home_own_idx],
    "loan_amnt": loan_amnt,
    "loan_intent": loan_intent_map[loan_intent_idx],
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person,
    "credit_score": credit_score,
    "previous_loan_defaults_on_file": prev_map[prev_idx],
}

df_input = pd.DataFrame([data])

if st.button(t["btn_analyze"], type="primary"):
    proba = model.predict_proba(df_input)[:, 1][0]

    if proba > 0.7:
        st.success(t["success_msg"].format(prob=100 * proba))
    elif proba > 0.4:
        st.warning(t["warning_msg"].format(prob=100 * proba))
    else:
        st.error(t["error_msg"].format(prob=100 * proba))