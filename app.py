import streamlit as st
import numpy as np
import joblib

# 🛠 Set page config at the very top
st.set_page_config(page_title="Epidemic Predictor", layout="centered")

# 🔁 Load Random Forest model
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

model = load_model()

# Simulated country/province structure
data = {
    "USA": ["California", "New York", "Texas"],
    "India": ["Maharashtra", "Delhi", "Kerala"],
    "Brazil": ["São Paulo", "Rio de Janeiro"],
    "Italy": ["Lombardy", "Lazio"]
}

# App content
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
