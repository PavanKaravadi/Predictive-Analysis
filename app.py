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
latest_data['Province_State'] = latest_data['Province_State'].fillna('')
latest_data['Country_Region'] = latest_data['Country_Region'].fillna('')

# Dropdown inputs
country = st.selectbox("Select Country", sorted(latest_data['Country_Region'].unique()))
states_filtered = latest_data[latest_data['Country_Region'] == country]['Province_State'].unique()
state = st.selectbox("Select State/Province", sorted(states_filtered))

# Select date
last_known_date = st.date_input("Last Known Case Date", datetime.date(2023, 1, 14))

# Prepare encoders
country_encoder = LabelEncoder()
state_encoder = LabelEncoder()
country_encoder.fit(latest_data['Country_Region'])
state_encoder.fit(latest_data['Province_State'])

# Predict Button
if st.button("Predict for Next 30 Days"):
    # Get historical data for selected region - fixed column name mismatch
    region_mask = (
        (confirmed_df['Country/Region'] == country) & 
        (confirmed_df['Province/State'].fillna('') == state)
    )
    
    # Safely get region cases
    if region_mask.any():
        region_cases = confirmed_df[region_mask].iloc[:, 4:].T
        region_cases = region_cases.rename(columns={region_cases.columns[0]: 'Cases'})
        region_cases.index = pd.to_datetime(region_cases.index)
    else:
        st.error(f"No data available for {state}, {country}")
        st.stop()
    
    # Feature engineering
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
        
        results = []
        latest_row = train_df.iloc[-1]
        current_cases = latest_row['Cases']
        
        # Get deaths data
        region_deaths = deaths_df[region_mask].iloc[:, 4:].T.iloc[-1, 0] if region_mask.any() else 0
        population = 1e6  # Replace with actual population data
        
        # Predict for each future day
        for i, date in enumerate(future_dates):
            # Prepare features
            day_features = {
                'day': date.day,
                'month': date.month,
                'day_of_week': date.dayofweek,
                'day_of_year': date.dayofyear,
                'week_of_year': date.isocalendar().week,
                'country': country_encoded,
                'state': state_encoded,
                'lag_7': train_df['Cases'].iloc[-7] if len(train_df) >= 7 else current_cases,
                'lag_14': train_df['Cases'].iloc[-14] if len(train_df) >= 14 else current_cases,
                'rolling_avg_7': train_df['Cases'].iloc[-7:].mean() if len(train_df) >= 7 else current_cases
            }
            
            # Predict cases
            try:
                predicted_cases = max(0, int(model.predict(pd.DataFrame([day_features]))[0]))
            except:
                predicted_cases = int(current_cases * (1.02 ** (i+1)))  # Fallback
            
            # Calculate rates with better scaling
            predicted_incidence = (predicted_cases / population) * 100000 if population > 0 else 0
            predicted_mortality = (region_deaths / predicted_cases) * 100 if predicted_cases > 0 else 0
            
            results.append({
                'Date': date.date(),
                'Predicted Cases': predicted_cases,
                'Incidence Rate (per 100k)': round(predicted_incidence, 2),
                'Mortality Rate (%)': round(predicted_mortality, 2)
            })
            
            current_cases = predicted_cases
        
        # Create and display results
        result_df = pd.DataFrame(results)
        st.subheader(f"30-Day Prediction for {state if state else 'National'}, {country}")
        
        # Improved visualization
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Predicted Cases", result_df['Predicted Cases'])
            st.line_chart(result_df.set_index('Date')['Predicted Cases'],
                         use_container_width=True)
        
        with col2:
            st.metric("Average Incidence Rate", 
                      f"{result_df['Incidence Rate']}")
            st.line_chart(result_df.set_index('Date')[['Incidence Rate']],
                         use_container_width=True)
        
        st.write("Detailed Predictions:")
        st.dataframe(result_df.style.format({
            'Predicted Cases': '{:,}',
            'Incidence Rate (per 100k)': '{:.2f}',
            'Mortality Rate (%)': '{:.4f}'
        }))
