#%%
import streamlit as st

pages = {
    "About the Project": [
        st.Page("about.py", title="What the project do?")
    ],
    "Model": [
        st.Page("predict.py", title="Predict Loan Approval")
    ]
}

pg = st.navigation(pages)
pg.run()