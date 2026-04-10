
# === FULL HYPERPARAMETER TUNING (Self-contained) ===

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import xgboost as xgb
import numpy as np

# Load data
df = pd.read_csv('train.csv')

# Features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_cols = ['Pclass', 'Sex', 'Embarked']

numerical_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Full XGBoost Pipeline
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb.XGBClassifier(random_state=42, eval_metric='logloss'))
])

# Hyperparameter grid
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__subsample': [0.8, 1.0]
}

# Randomized Search
random_search = RandomizedSearchCV(
    xgb_pipeline,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

# Run tuning
random_search.fit(X, y)

print("Best Parameters:", random_search.best_params_)
print(f"Best Cross-Validation Accuracy: {random_search.best_score_:.4f}")

import joblib
best_model = random_search.best_estimator_
joblib.dump(best_model, 'titanic_best_model.pkl')
print("Model saved successfully")

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# 
import streamlit as st
import pandas as pd
import joblib

 # Load model
model = joblib.load('titanic_best_model.pkl')

st.title("🚢 Titanic Survival Predictor")
st.subheader("Will this passenger survive the Titanic?")

col1, col2 = st.columns(2)

with col1:
     pclass = st.selectbox("Passenger Class", [1, 2, 3])
     sex = st.selectbox("Sex", ["male", "female"])
     age = st.slider("Age", 0, 80, 30)
 
with col2:
     sibsp = st.number_input("Siblings/Spouses aboard", 0, 8, 0)
     parch = st.number_input("Parents/Children aboard", 0, 6, 0)
     fare = st.number_input("Fare paid ($)", 0.0, 500.0, 32.0)
     embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
 
input_data = pd.DataFrame({
     'Pclass': [pclass],
     'Sex': [sex],
     'Age': [age],
     'SibSp': [sibsp],
     'Parch': [parch],
     'Fare': [fare],
     'Embarked': [embarked]
 })
 
 if st.button("🔮 Predict Survival"):
     prediction = model.predict(input_data)[0]
     probability = model.predict_proba(input_data)[0][1]
 
     if prediction == 1:
         st.success(f"✅ Survived with probability {probability:.1%}")
     else:
         st.error(f"❌ Did not survive (survival probability: {probability:.1%})")
 
st.caption("Built with XGBoost + Streamlit | FELO's Data Science Portfolio")

