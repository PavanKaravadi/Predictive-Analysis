import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# App Config
st.set_page_config(page_title="COVID-19 Case Predictor", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("covid19_cases_predictor.pkl")

model = load_model()

# Sample country/state data
region_data = {
    "USA": ["California", "New York", "Texas"],
    "India": ["Maharashtra", "Delhi", "Kerala"],
    "Brazil": ["São Paulo", "Rio de Janeiro"],
    "Italy": ["Lombardy", "Lazio"]
}

# Dummy historical data - In real case, you’d load actual recent country-wise case data
@st.cache_data
def load_historical_data():
    dates = pd.date_range(end=pd.Timestamp.today(), periods=100)
    data = pd.DataFrame({
        "date": dates,
        "world_cases": np.linspace(100, 50000, 100) + np.random.randint(-1000, 1000, 100)
    })
    data.set_index("date", inplace=True)
    return data

historical_data = load_historical_data()

# Dummy update_features logic — customize to match your feature engineering
def update_features(current_features, last_prediction, current_date):
    # Example: assume lag feature is just last prediction scaled
    return [last_prediction * 1.01]  # update with your actual logic

def predict_future_cases(model, last_known_date, num_days, initial_features):
    predictions = []
    current_features = initial_features.copy()
    current_date = last_known_date

    for _ in range(num_days):
        pred = model.predict([current_features])[0]
        predictions.append(pred)
        current_date += pd.Timedelta(days=1)
        current_features = update_features(current_features, pred, current_date)

    return pd.DataFrame({
        "date": pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=num_days),
        "predicted_cases": predictions
    }).set_index("date")

# UI Elements
st.title("🦠 COVID-19 Cases Predictor (Next 30 Days)")
country = st.selectbox("🌐 Select Country", list(region_data.keys()))
state = st.selectbox("🏙️ Select State", region_data[country])
start_date = st.date_input("📅 Start Prediction From", datetime.today())

# Initial features to kick off prediction (simulated)
initial_features = [historical_data["world_cases"].iloc[-1]]

if st.button("🔮 Predict"):
    # Predict next 30 days
    last_known_date = historical_data.index[-1]
    forecast = predict_future_cases(model, last_known_date, 30, initial_features)

    # Combine with historical data
    combined = pd.concat([historical_data.tail(60)["world_cases"], forecast["predicted_cases"]])
    st.line_chart(combined.rename("Confirmed Cases"))

    st.success(f"✅ Prediction completed for {state}, {country} from {start_date.strftime('%Y-%m-%d')}")
    st.dataframe(forecast.reset_index().rename(columns={"date": "Date", "predicted_cases": "Predicted Cases"}))
