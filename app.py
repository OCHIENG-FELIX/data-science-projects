import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn

# Load the saved model
model = joblib.load('titanic_best_model.pkl')

st.title("🚢 Titanic Survival Predictor")
st.subheader("Will this passenger survive?")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 30)

with col2:
    sibsp = st.number_input("Siblings / Spouses aboard", 0, 8, 0)
    parch = st.number_input("Parents / Children aboard", 0, 6, 0)
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
