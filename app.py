import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle


with open("models/random_forest_mall.sav", "rb") as f:
    model = pickle.load(f)


st.title("Spending Score - Model Prediction")


val1 = st.slider("Sex (0: Female, 1: Male)", min_value=0.0, max_value=1.0, step=1.0)
val2 = st.slider("Age", min_value=0.0, max_value=100.0, step=1.0)
val3 = st.slider("Annual Income (in $)", min_value=0.0, max_value=200000.0, step=500.0)


if st.button("Predict"):
    prediction = str(model.predict([[val1, val2, val3]])[0])
    #pred_class = class_dict[prediction]
    st.write("Prediction:", prediction)
