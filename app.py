import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np

# Load ML model
model = joblib.load("covid19_cases_predictor.pkl")

# Load datasets
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/01-15-2023.csv')

# Title
st.title("COVID-19 Prediction Dashboard")

# Preprocess location info from latest data
latest_data['Province_State'] = latest_data['Province_State'].fillna('Unknown')
latest_data['Country_Region'] = latest_data['Country_Region'].fillna('Unknown')

# Dropdown inputs
country = st.selectbox("Select Country", sorted(latest_data['Country_Region'].unique()))
states_filtered = latest_data[latest_data['Country_Region'] == country]['Province_State'].unique()
state = st.selectbox("Select State", sorted(states_filtered))

# Select date
last_known_date = st.date_input("Last Known Case Date", datetime.date.today())

# Predict Button
if st.button("Predict for Next 30 Days"):
    # Encode country and state (replace with actual encodings used in your model)
    country_encoded = hash(country) % 1000
    state_encoded = hash(state) % 1000

    # Prepare future dates
    future_days = np.arange(1, 31)
    future_dates = [last_known_date + datetime.timedelta(days=int(i)) for i in future_days]

    predicted_cases = []
    incidence_rates = []
    mortality_rates = []

    # Get real values from latest report for this country and state
    try:
        region_row = latest_data[
            (latest_data['Country_Region'] == country) & 
            (latest_data['Province_State'] == state)
        ].iloc[0]

        confirmed = region_row['Confirmed']
        deaths = region_row['Deaths']
        incident_rate_base = region_row['Incident_Rate'] if region_row['Incident_Rate'] > 0 else 0
        fatality_ratio_base = region_row['Case_Fatality_Ratio'] if region_row['Case_Fatality_Ratio'] > 0 else 0
    except:
        confirmed, deaths, incident_rate_base, fatality_ratio_base = 1000, 10, 5.0, 1.0

    # Predict for each day
    for day in future_days:
        features = [[day, country_encoded, state_encoded]]
        try:
            pred = model.predict(features)[0]
        except Exception as e:
            pred = 0
        pred = max(0, int(pred))
        predicted_cases.append(pred)

        # Calculate rates (example logic)
        predicted_incidence = incident_rate_base + (pred / 100000 * 100)  # fake increment
        predicted_mortality = (deaths + pred * fatality_ratio_base / 100) / max(1, confirmed + pred) * 100

        incidence_rates.append(round(predicted_incidence, 2))
        mortality_rates.append(round(predicted_mortality, 2))

    # Output DataFrame
    result_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Confirmed Cases': predicted_cases,
        'Estimated Incidence Rate (per 100k)': incidence_rates,
        'Estimated Mortality Rate (%)': mortality_rates
    })

    # Show results
    st.subheader("Prediction Results")
    st.dataframe(result_df)
    st.line_chart(result_df.set_index('Date')[['Predicted Confirmed Cases']])
    st.line_chart(result_df.set_index('Date')[['Estimated Incidence Rate (per 100k)', 'Estimated Mortality Rate (%)']])
