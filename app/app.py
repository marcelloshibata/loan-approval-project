#%%
import streamlit as st

pages = {
    "Sobre o Projeto": [
        st.Page("about.py", title="Detalhes")
    ],
    "Model": [
        st.Page("predict.py", title="Predict Loan Approval")
    ]
}

pg = st.navigation(pages)
pg.run()