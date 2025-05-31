import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load ML model
@st.cache_resource
def load_model():
    try:
        return joblib.load("covid19_cases_predictor.pkl")
    except Exception as e:
        st.warning(f"Model loading failed: {str(e)}. Using dummy model.")
        return RandomForestRegressor(random_state=42)

model = load_model()

# Load datasets
@st.cache_data
def load_data():
    confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/01-15-2023.csv')
    return confirmed_df, deaths_df, latest_data

confirmed_df, deaths_df, latest_data = load_data()

# Title
st.title("COVID-19 Daily Cases Prediction Dashboard")

# Preprocess location info
latest_data['Province_State'] = latest_data['Province_State'].fillna('')
latest_data['Country_Region'] = latest_data['Country_Region'].fillna('')

# Dropdown inputs
country = st.selectbox("Select Country", sorted(latest_data['Country_Region'].unique()))
states_filtered = latest_data[latest_data['Country_Region'] == country]['Province_State'].unique()
state = st.selectbox("Select State/Province", ['National'] + sorted([s for s in states_filtered if s != '']))

# Date selection
last_known_date = st.date_input("Last Known Case Date", datetime.date(2023, 1, 14))

# Prepare encoders
country_encoder = LabelEncoder()
state_encoder = LabelEncoder()
country_encoder.fit(latest_data['Country_Region'])
state_encoder.fit(['National'] + latest_data['Province_State'].dropna().unique().tolist())

# Predict Button
if st.button("Predict for Next 30 Days"):
    # Get historical data
    if state == 'National':
        region_mask = (confirmed_df['Country/Region'] == country)
    else:
        region_mask = ((confirmed_df['Country/Region'] == country) & 
                      (confirmed_df['Province/State'].fillna('') == state))
    
    if not region_mask.any():
        st.error("No data available for selected region")
        st.stop()
    
    # Get cases and convert to daily (non-cumulative)
    region_cases = confirmed_df[region_mask].iloc[:, 4:].T
    region_cases = region_cases.diff(axis=0).dropna()  # Convert to daily new cases
    region_cases.columns = ['Daily Cases']
    region_cases.index = pd.to_datetime(region_cases.index)
    region_cases = region_cases.clip(lower=0)  # Remove negative values
    
    # Feature engineering
    def create_features(df):
        df = df.copy()
        # Date features
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        
        # Lag features
        for lag in [1, 2, 3, 7, 14]:
            df[f'lag_{lag}'] = df['Daily Cases'].shift(lag).fillna(method='ffill')
        
        # Rolling features
        df['rolling_avg_7'] = df['Daily Cases'].rolling(7, min_periods=1).mean()
        df['rolling_std_7'] = df['Daily Cases'].rolling(7, min_periods=1).std()
        
        return df.dropna(subset=['Daily Cases'])

    train_df = create_features(region_cases)
    
    if len(train_df) < 14:  # Need at least 2 weeks of data
        st.error("Insufficient historical data for prediction")
        st.stop()
    
    # Encode location
    try:
        country_encoded = country_encoder.transform([country])[0]
        state_encoded = state_encoder.transform([state])[0] if state != 'National' else 0
    except ValueError as e:
        st.error(f"Encoding error: {str(e)}")
        st.stop()
    
    # Prepare features for prediction
    feature_cols = [col for col in train_df.columns if col != 'Daily Cases']
    
    # Predict next 30 days
    predictions = []
    current_features = train_df.iloc[-1][feature_cols].copy()
    
    for i in range(30):
        # Update date features for the prediction day
        pred_date = last_known_date + datetime.timedelta(days=i+1)
        current_features['day'] = pred_date.day
        current_features['month'] = pred_date.month
        current_features['day_of_week'] = pred_date.weekday()
        current_features['day_of_year'] = pred_date.timetuple().tm_yday
        current_features['country'] = country_encoded
        current_features['state'] = state_encoded
        
        # Prepare input features (must match model training)
        input_features = pd.DataFrame([current_features])[feature_cols]
        
        # Make prediction
        try:
            pred = max(0, int(model.predict(input_features)[0]))
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            pred = 0
        
        predictions.append({
            'Date': pred_date,
            'Predicted Daily Cases': pred,
            'Incidence Rate (per 100k)': round((pred / 1e6) * 100000, 2) if pred > 0 else 0,
            'Mortality Rate (%)': round((deaths_df[region_mask].iloc[:, 4:].T.iloc[-1, 0] / pred * 100, 4)) if pred > 0 else 0
        })
        
        # Update features for next prediction
        for lag in [14, 7, 3, 2, 1]:
            if lag == 1:
                current_features[f'lag_{lag}'] = pred
            else:
                current_features[f'lag_{lag}'] = current_features[f'lag_{lag-1}']
        
        current_features['rolling_avg_7'] = np.mean([current_features[f'lag_{l}'] for l in range(1, 8)])
        current_features['rolling_std_7'] = np.std([current_features[f'lag_{l}'] for l in range(1, 8)])
    
    # Create results
    result_df = pd.DataFrame(predictions).set_index('Date')
    
    # Show results
    st.subheader(f"Daily Case Predictions for {state if state != 'National' else ''} {country}".strip())
    
    # Plot historical vs predicted
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Daily Cases", f"{result_df['Predicted Daily Cases'].mean():,.0f}")
        st.line_chart(result_df['Predicted Daily Cases'], use_container_width=True)
    
    with col2:
        st.metric("Total Predicted Cases", f"{result_df['Predicted Daily Cases'].sum():,.0f}")
        st.line_chart(result_df[['Incidence Rate (per 100k)', 'Mortality Rate (%)']], use_container_width=True)
    
    # Detailed data
    st.write("Detailed Predictions:")
    st.dataframe(
        result_df.style.format({
            'Predicted Daily Cases': '{:,}',
            'Incidence Rate (per 100k)': '{:.2f}',
            'Mortality Rate (%)': '{:.4f}'
        }).background_gradient(cmap='YlOrRd'),
        use_container_width=True
    )

# Instructions
st.markdown("""
**Instructions:**
1. Select country and state/province (or 'National' for country-level)
2. Set the last known case date
3. Click "Predict for Next 30 Days"

**Note:** Shows daily new cases, not cumulative totals.
""")
