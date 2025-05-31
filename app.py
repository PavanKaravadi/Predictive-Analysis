import streamlit as st
import numpy as np
import joblib

# Load model trained and saved in the SAME environment
model = joblib.load("random_forest_model.pkl")

# Simulated country/province data
data = {
    "USA": ["California", "New York", "Texas"],
    "India": ["Maharashtra", "Delhi", "Kerala"],
    "Brazil": ["São Paulo", "Rio de Janeiro"],
    "Italy": ["Lombardy", "Lazio"]
}

# UI
st.set_page_config(page_title="Epidemic Predictor", layout="centered")

st.markdown("""
# 🦠 Epidemic Outbreak Predictor
### 🌍 Country-wise Prediction | Powered by Random Forest

This app forecasts the number of confirmed COVID-19 cases based on user input and region.

---
""")

# Dropdowns
country = st.selectbox("🌐 Select Country", list(data.keys()))
province = st.selectbox("🏙️ Select Province/State", data[country])

# Input
days = st.number_input("📅 Enter number of days since outbreak began:", min_value=1, max_value=1000)

# Button
if st.button("🔮 Predict Confirmed Cases"):
    input_array = np.array([[days]])
    prediction = model.predict(input_array)
    st.success(f"🧾 Predicted Confirmed Cases on Day {days} in **{province}, {country}**: **{int(prediction[0]):,}**")
