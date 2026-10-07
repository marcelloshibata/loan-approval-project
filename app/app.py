#%%
import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5050/")
models = mlflow.search_registered_models(filter_string="name = 'model_pr'")
latest_version = max([i.version for i in models[0].latest_versions])

model = mlflow.sklearn.load_model(f"models:/model_pr/{latest_version}")
features = model.feature_names_in_

df = pd.read_csv("../data/loan_data.csv")
amostra = df.sample()
amostra = amostra[features]

# %%
predict = model.predict_proba(amostra)
amostra['proba'] = predict[:,1]
amostra