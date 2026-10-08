#%%
import streamlit as st

pages = {
    "About the Project": [
        st.Page("about.py", title="Details")
    ],
    "Model": [
        st.Page("predict.py", title="Predict Loan Approval")
    ]
}

pg = st.navigation(pages)
pg.run()