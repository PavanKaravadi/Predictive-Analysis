import streamlit as st
import pandas as pd
import joblib
import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load ML model
try:
    model = joblib.load("covid19_cases_predictor.pkl")
except:
    st.warning("Model file not found. Using a dummy model for demonstration.")
    model = RandomForestRegressor(random_state=42)  # Fallback dummy model with fixed seed

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
state = st.selectbox("Select State/Province", ['National'] + sorted([s for s in states_filtered if s != '']))

# Select date
last_known_date = st.date_input("Last Known Case Date", datetime.date(2023, 1, 14))

# Prepare encoders
country_encoder = LabelEncoder()
state_encoder = LabelEncoder()
country_encoder.fit(latest_data['Country_Region'])
state_encoder.fit(['National'] + latest_data['Province_State'].dropna().unique().tolist())

# Predict Button
if st.button("Predict for Next 30 Days"):
    # Get historical data for selected region
    if state == 'National':
        region_mask = (confirmed_df['Country/Region'] == country)
    else:
        region_mask = ((confirmed_df['Country/Region'] == country) & 
                      (confirmed_df['Province/State'].fillna('') == state))
    
    if not region_mask.any():
        st.error("No data available for selected region")
        st.stop()
    
    region_cases = confirmed_df[region_mask].iloc[:, 4:].T
    region_cases = region_cases.rename(columns={region_cases.columns[0]: 'Cases'})
    region_cases.index = pd.to_datetime(region_cases.index)
    
    # Feature engineering with proper null handling
    def create_features(df):
        df = df.copy()
        
        # Basic date features
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        
        # Lag features with forward fill
        for lag in [1, 2, 3, 7, 14]:
            df[f'lag_{lag}'] = df['Cases'].shift(lag).ffill()
        
        # Rolling features with min_periods
        df['rolling_avg_7'] = df['Cases'].rolling(7, min_periods=1).mean()
        df['rolling_std_7'] = df['Cases'].rolling(7, min_periods=1).std()
        
        return df.dropna(subset=['Cases'])

    train_df = create_features(region_cases)
    
    if len(train_df) == 0:
        st.error("No valid historical data available for prediction")
        st.stop()
    
    # Encode location features safely
    try:
        country_encoded = country_encoder.transform([country])[0]
        state_encoded = state_encoder.transform([state])[0] if state != 'National' else 0
    except ValueError as e:
        st.error(f"Encoding error: {str(e)}")
        st.stop()
    
    # Prepare future dates
    future_dates = pd.date_range(
        start=last_known_date + datetime.timedelta(days=1),
        periods=30
    )
    
    # Prepare empty results DataFrame
    results = []
    current_features = train_df.iloc[-1].to_dict()
    
    # Imputer for handling missing values
    imputer = SimpleImputer(strategy='mean')
    feature_cols = [col for col in train_df.columns if col != 'Cases']
    X_train = imputer.fit_transform(train_df[feature_cols])
    
    # Train model (for demo - in production, use pre-trained)
    try:
        model.fit(X_train, train_df['Cases'])
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        st.stop()
    
    # Predict for each future day
    for i, date in enumerate(future_dates):
        # Update date features
        current_features.update({
            'day': date.day,
            'month': date.month,
            'day_of_week': date.dayofweek,
            'day_of_year': date.dayofyear,
            'week_of_year': date.isocalendar().week,
            'country': country_encoded,
            'state': state_encoded
        })
        
        # Prepare feature vector
        feature_vector = np.array([[current_features[col] for col in feature_cols]])
        feature_vector = imputer.transform(feature_vector)
        
        # Predict cases
        try:
            predicted_cases = max(0, int(model.predict(feature_vector)[0]))
        except Exception as e:
            st.warning(f"Prediction failed for day {i+1}: {str(e)}")
            predicted_cases = 0
        
        # Store results
        results.append({
            'Date': date.date(),
            'Predicted Daily Cases': predicted_cases
        })
        
        # Update features for next prediction
        for lag in [14, 7, 3, 2, 1]:
            if f'lag_{lag}' in current_features:
                if lag == 1:
                    current_features[f'lag_{lag}'] = predicted_cases
                else:
                    current_features[f'lag_{lag}'] = current_features[f'lag_{lag-1}']
        
        current_features['rolling_avg_7'] = np.mean([
            current_features.get(f'lag_{l}', 0) for l in range(1, 8)
        ])
    
    # Create results DataFrame
    result_df = pd.DataFrame(results).set_index('Date')
    
    # Show results
    st.subheader(f"30-Day Daily Cases Prediction for {state if state != 'National' else ''} {country}".strip())
    
    # Plotting
    col1, col2 = st.columns(2)
    
    with col1:
        st.line_chart(result_df['Predicted Daily Cases'])
    
    with col2:
        st.dataframe(
            result_df.style.format("{:,.0f}").background_gradient(cmap='YlOrRd'),
            use_container_width=True
        )
    
    # Add some metrics
    st.metric("Average Predicted Daily Cases", f"{result_df['Predicted Daily Cases'].mean():,.0f}")
    st.metric("Total Predicted Cases", f"{result_df['Predicted Daily Cases'].sum():,.0f}")

# Add some instructions
st.markdown("""
**Note:** 
- Predictions show daily new cases (not cumulative totals)
- 'National' will show country-level predictions
- For states with no data, predictions may be less accurate
""")
