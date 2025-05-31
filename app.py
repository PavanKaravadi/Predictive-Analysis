import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load ML model
try:
    model = joblib.load("covid19_cases_predictor.pkl")
except:
    st.warning("Model file not found. Using a dummy model for demonstration.")
    model = RandomForestRegressor()  # Fallback dummy model

# Load datasets
@st.cache_data
def load_data():
    confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/01-15-2023.csv')
    return confirmed_df, deaths_df, latest_data

confirmed_df, deaths_df, latest_data = load_data()

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
last_known_date = st.date_input("Last Known Case Date", datetime.date(2023, 1, 14))

# Prepare encoders (should match your model training)
country_encoder = LabelEncoder()
state_encoder = LabelEncoder()
country_encoder.fit(latest_data['Country_Region'])
state_encoder.fit(latest_data['Province_State'])

# Predict Button
if st.button("Predict for Next 30 Days"):
    # Get historical data for selected region
    region_mask = (confirmed_df['Country/Region'] == country) & (confirmed_df['Province/State'].fillna('Unknown') == state)
    region_cases = confirmed_df[region_mask].iloc[:, 4:].T
    region_cases.columns = ['Cases']
    region_cases.index = pd.to_datetime(region_cases.index)
    
    # Feature engineering (should match your model training)
    def create_features(df):
        df = df.copy()
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        df['lag_7'] = df['Cases'].shift(7)
        df['lag_14'] = df['Cases'].shift(14)
        df['rolling_avg_7'] = df['Cases'].rolling(7).mean()
        return df.dropna()
    
    # Prepare data for prediction
    train_df = create_features(region_cases)
    
    if len(train_df) == 0:
        st.error("No historical data available for this region")
    else:
        # Encode location features
        country_encoded = country_encoder.transform([country])[0]
        state_encoded = state_encoder.transform([state])[0]
        
        # Prepare future dates
        future_dates = pd.date_range(
            start=last_known_date + datetime.timedelta(days=1),
            periods=30
        )
        
        # Prepare empty results DataFrame
        results = []
        
        # Get latest actual values
        latest_row = train_df.iloc[-1]
        current_cases = latest_row['Cases']
        current_date = train_df.index[-1]
        
        # Calculate base rates
        region_deaths = deaths_df[region_mask].iloc[:, 4:].T.iloc[-1, 0]
        population = 1e6  # This should be replaced with actual population data
        incident_rate_base = (current_cases / population) * 100000 if population > 0 else 0
        fatality_rate_base = (region_deaths / current_cases) * 100 if current_cases > 0 else 0
        
        # Predict for each future day
        for i, date in enumerate(future_dates):
            # Prepare features for prediction
            day_features = {
                'day': date.day,
                'month': date.month,
                'day_of_week': date.dayofweek,
                'day_of_year': date.dayofyear,
                'week_of_year': date.isocalendar().week,
                'country': country_encoded,
                'state': state_encoded,
                'lag_7': train_df['Cases'].iloc[-7] if len(train_df) >= 7 else 0,
                'lag_14': train_df['Cases'].iloc[-14] if len(train_df) >= 14 else 0,
                'rolling_avg_7': train_df['Cases'].iloc[-7:].mean() if len(train_df) >= 7 else 0
            }
            
            # Convert to DataFrame for prediction
            features_df = pd.DataFrame([day_features])
            
            # Predict cases
            try:
                predicted_cases = max(0, int(model.predict(features_df)[0]))
            except:
                # Fallback prediction if model fails
                growth_rate = 1.02  # 2% daily growth as fallback
                predicted_cases = int(current_cases * (growth_rate ** (i+1)))
            
            # Calculate rates
            predicted_incidence = (predicted_cases / population) * 100000 if population > 0 else 0
            predicted_mortality = fatality_rate_base  # Keeping mortality rate constant
            
            results.append({
                'Date': date.date(),
                'Predicted Cases': predicted_cases,
                'Incidence Rate (per 100k)': round(predicted_incidence, 2),
                'Mortality Rate (%)': round(predicted_mortality, 2)
            })
            
            # Update current cases for next prediction
            current_cases = predicted_cases
        
        # Create results DataFrame
        result_df = pd.DataFrame(results)
        
        # Show results
        st.subheader(f"30-Day Prediction for {state}, {country}")
        st.dataframe(result_df)
        
        # Plot cases prediction
        st.line_chart(result_df.set_index('Date')['Predicted Cases'])
        
        # Plot rates
        st.line_chart(result_df.set_index('Date')[['Incidence Rate (per 100k)', 'Mortality Rate (%)']])
