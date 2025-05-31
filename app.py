import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model (still using the general polynomial model for now)
model = joblib.load("polynomial_regression_model.pkl")

# Simulated country/province structure (replace this with your real data later)
data = {
    "USA": ["California", "New York", "Texas"],
    "India": ["Maharashtra", "Delhi", "Kerala"],
    "Brazil": ["São Paulo", "Rio de Janeiro"],
    "Italy": ["Lombardy", "Lazio"]
}

# App title and description
st.markdown("""
# 🦠 Epidemic Outbreak Predictor
### 🌍 Country-wise Prediction | Powered by Polynomial Regression

This app forecasts the number of confirmed COVID-19 cases based on user input and region.

---

📈 **Model**: Polynomial Regression  
🧠 **Tech**: Streamlit + Scikit-learn  
""")

# Country Selection
country = st.selectbox("🌐 Select Country", list(data.keys()))

# Province/State Selection (optional based on country)
province = st.selectbox("🏙️ Select Province/State", data[country])

# Number of days since outbreak began
days = st.number_input("📅 Enter number of days since outbreak began:", min_value=1, max_value=1000)

# Predict button
if st.button("🔮 Predict Confirmed Cases"):
    input_array = np.array([[days]])
    prediction = model.predict(input_array)

    st.success(f"🧾 Predicted Confirmed Cases on Day {days} in **{province}, {country}**: **{int(prediction[0]):,}**")