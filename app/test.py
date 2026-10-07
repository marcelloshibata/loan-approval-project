#%%
import pandas as pd

df = pd.read_csv("../data/loan_data.csv")
df.head(20)

# %%
st = "Debt Consolidation"

print(st.replace(" ", "").upper())