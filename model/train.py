#%%
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import tree

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
encoder = preprocessing.OrdinalEncoder()
text_columns = X_train.select_dtypes(include=['object']).columns
text_columns
X_train[text_columns] = encoder.fit_transform(X_train[text_columns])
X_test[text_columns] = encoder.fit_transform(X_test[text_columns])

# %% deciding what is the most important features for target
tree_feat = tree.DecisionTreeClassifier(random_state=42)
tree_feat.fit(X_train, y_train)

feature_importances = (pd.Series(tree_feat.feature_importances_,
index=X_train.columns).sort_values(ascending=False).reset_index())
feature_importances['acum.'] = feature_importances[0].cumsum()
feature_importances[feature_importances['acum.'] < 0.96]

best_features = (feature_importances[feature_importances['acum.'] < 0.96]['index'].tolist())
best_features

# %%
