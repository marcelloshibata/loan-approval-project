#%%
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import tree
from feature_engine import discretisation
from sklearn import tree
from sklearn import metrics
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import pipeline
import matplotlib.pyplot as plt
from sklearn import set_config
from sklearn.compose import ColumnTransformer
import numpy as np
import mlflow

set_config(transform_output="pandas")

mlflow.set_tracking_uri("http://127.0.0.1:5050/")
mlflow.set_experiment(experiment_id=1)

df = pd.read_csv("../data/loan_data.csv")
df.head()

target = 'loan_status'

X = df.iloc[:, :-1]
y = df[target]

# %% Sample
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    random_state=42,
                                                                    test_size=0.2,
                                                                    stratify=y,)

print("Taxa variavel resposta Treino: ", y_train.mean())
print("Taxa variavel resposta Teste: ", y_test.mean())

# %% EXPLORE - transforming text values in numbers
encoder = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',
                                       unknown_value=np.nan)

X_train_temp = X_train.copy()

text_columns = X_train_temp.select_dtypes(include=['object']).columns
text_columns
X_train_temp[text_columns] = encoder.fit_transform(X_train_temp[text_columns])

# %% deciding what is the most important features for target
tree_feat = tree.DecisionTreeClassifier(random_state=42)
tree_feat.fit(X_train_temp, y_train)

feature_importances = (pd.Series(tree_feat.feature_importances_,
index=X_train_temp.columns).sort_values(ascending=False).reset_index())
feature_importances['acum.'] = feature_importances[0].cumsum()
feature_importances[feature_importances['acum.'] < 0.96]

best_features = (feature_importances[feature_importances['acum.'] < 0.96]['index'].tolist())
best_features

# %%
from feature_engine.encoding import OrdinalEncoder
from feature_engine.selection import DropFeatures

features_to_drop = [col for col in X_train.columns if col not in best_features]

text_columns_in_best = [col for col in best_features if col in X_train.select_dtypes(include=['object', 'category']).columns]
encoder = OrdinalEncoder(encoding_method='arbitrary', variables=text_columns_in_best)

num_columns_in_best = [col for col in best_features if col not in text_columns_in_best]
tree_discretisation = discretisation.DecisionTreeDiscretiser(
    variables=num_columns_in_best,
    regression=False,
    bin_output='bin_number',
    cv=3
)

# model = linear_model.LogisticRegression(
#     penalty=None,
#     random_state=42,
#     max_iter=10000
# )

# model = naive_bayes.BernoulliNB()

# model = ensemble.AdaBoostClassifier(
#     random_state=42,
#     n_estimators=800,
#     learning_rate=0.01
# )

#best model
model = ensemble.RandomForestClassifier(
    random_state=42,
    n_jobs=2,
)

params = {
    "min_samples_leaf":[50, 60, 80, 70, 100],
    "n_estimators":[700, 600, 500, 1000, 1300],
    "criterion": ['gini', 'entropy', 'log_loss'],
}

grid = model_selection.GridSearchCV(
    model, params, cv=3, scoring='roc_auc',
    verbose=4)

model_pipeline = pipeline.Pipeline(
    steps=[
        ('Drop_Extra_Features', DropFeatures(features_to_drop=features_to_drop)),   
        ('Encoder', encoder),
        ('Discretiser', tree_discretisation),
        ('Grid', grid)
    ]
)

with mlflow.start_run(run_name=model.__str__()):
    mlflow.sklearn.autolog()
    model_pipeline.fit(X_train, y_train)

    # ASSESS
    y_train_predict = model_pipeline.predict(X_train)
    y_train_proba = model_pipeline.predict_proba(X_train)[:,1]

    acc_train = metrics.accuracy_score(y_train, y_train_predict)
    auc_train = metrics.roc_auc_score(y_train, y_train_proba)
    roc_train = metrics.roc_curve(y_train, y_train_proba)
    print("Train accuracy: ", acc_train)
    print("Train AUC: ", auc_train)

    # now in test base
    y_test_predict = model_pipeline.predict(X_test)
    y_test_proba = model_pipeline.predict_proba(X_test)[:,1]
    acc_test = metrics.accuracy_score(y_test, y_test_predict)
    auc_test = metrics.roc_auc_score(y_test, y_test_proba)
    roc_test = metrics.roc_curve(y_test, y_test_proba)
    print("Test accuracy: ", acc_test)
    print("Test AUC: ", auc_test)

    mlflow.log_metrics({
        "acc_train":acc_train,
        "auc_train":auc_train,
        "acc_test":acc_test,
        "auc_test":auc_test,
    })

# %%
plt.figure(dpi=400)
plt.plot(roc_train[0], roc_train[1])
plt.plot(roc_test[0], roc_test[1])
plt.grid(True)
plt.title("ROC Curve")
plt.legend([
    f"Train: {100*auc_train:.2f}",
    f"Test: {100*auc_test:.2f}"
])
plt.show()