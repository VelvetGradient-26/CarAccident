import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

st.set_page_config(page_title="Accident Severity Predictor", layout="wide")

st.title("🚗 Accident Severity Prediction")
st.write("Predict severity based on accident conditions")


model = joblib.load("models/lgbm_model.pkl")
le = joblib.load("models/label_encoder.pkl")
weights = joblib.load("models/threshold_weights.pkl")

# ---------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------
st.sidebar.header("Input Features")

def user_input():
    data = {
        "Day_of_week": st.sidebar.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]),
        "Age_band_of_driver": st.sidebar.selectbox("Driver Age", ["18-30","31-50","51+"]),
        "Sex_of_driver": st.sidebar.selectbox("Driver Sex", ["Male","Female"]),
        "Driving_experience": st.sidebar.selectbox("Experience", ["<1yr","1-5yr","5+"]),
        "Type_of_vehicle": st.sidebar.selectbox("Vehicle", ["Car","Truck","Motorcycle"]),
        "Area_accident_occured": st.sidebar.selectbox("Area", ["Urban","Rural"]),
        "Road_surface_conditions": st.sidebar.selectbox("Road Condition", ["Dry","Wet"]),
        "Light_conditions": st.sidebar.selectbox("Light", ["Daylight","Night"]),
        "Weather_conditions": st.sidebar.selectbox("Weather", ["Clear","Rainy"]),
        "Type_of_collision": st.sidebar.selectbox("Collision", ["Rear","Head-on","Side"]),
        "Number_of_vehicles_involved": st.sidebar.slider("Vehicles", 1, 5, 2),
        "Number_of_casualties": st.sidebar.slider("Casualties", 0, 10, 1),
        "Hour_of_Day": st.sidebar.slider("Hour", 0, 23, 12),
        "Time_of_Day": st.sidebar.selectbox("Time of Day", ["Morning","Afternoon","Evening","Night"]),
    }
    return pd.DataFrame([data])

input_df = user_input()

# ---------------------------------------------------------
# PREPROCESS (MATCH TRAINING)
# ---------------------------------------------------------
def preprocess(df):
    df = pd.get_dummies(df)

    # align columns with training
    train_cols = model.feature_name_
    df = df.reindex(columns=train_cols, fill_value=0)

    return df

X = preprocess(input_df)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if st.button("Predict"):
    proba = model.predict_proba(X)

    # Apply weights
    preds = np.argmax(proba * weights, axis=1)

    # Rule overrides
    if proba[0][0] > 0.07:
        preds[0] = 0

    if proba[0][1] > 0.18 and preds[0] != 0:
        preds[0] = 1

    label = le.inverse_transform(preds)[0]

    st.subheader("Prediction:")
    st.success(label)

    st.subheader("Probabilities:")
    st.write(proba)