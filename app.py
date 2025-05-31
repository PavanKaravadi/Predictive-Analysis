import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris  # Replace with real data

# ✅ This must be the first Streamlit command
st.set_page_config(page_title="Epidemic Predictor", layout="centered")

# Train model inside the app (to avoid version mismatch)
@st.cache_resource
def train_model():
    # Replace this with your actual training dataset
    X, y = load_iris(return_X_y=True)
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# Simulated country/province data
data = {
    "USA": ["California", "New York", "Texas"],
    "India": ["Maharashtra", "Delhi", "Kerala"],
    "Brazil": ["São Paulo", "Rio de Janeiro"],
    "Italy": ["Lombardy", "Lazio"]
}

# UI Content
st.markdown("""
# 🦠 Epidemic Outbreak Predictor
### 🌍 Country-wise Prediction | Powered by Random Forest

This app forecasts the number of confirmed COVID-19 cases based on user input and region.

---
""")

country = st.selectbox("🌐 Select Country", list(data.keys()))
province = st.selectbox("🏙️ Select Province/State", data[country])
days = st.number_input("📅 Enter number of days since outbreak began:", min_value=1, max_value=1000)

if st.button("🔮 Predict Confirmed Cases"):
    input_array = np.array([[days]])
    prediction = model.predict(input_array)
    st.success(f"🧾 Predicted Confirmed Cases on Day {days} in **{province}, {country}**: **{int(prediction[0]):,}**")
