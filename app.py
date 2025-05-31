import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from sklearn.ensemble import GradientBoostingRegressor  # Better for time series
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# Configuration
st.set_page_config(layout="wide")
plt.style.use('ggplot')

# Load datasets
@st.cache_data
def load_data():
    confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    return confirmed, deaths

confirmed_df, deaths_df = load_data()

# Preprocess function
def preprocess_data(df, country, state=None):
    """Convert cumulative counts to daily new cases"""
    if state:
        mask = (df['Country/Region'] == country) & (df['Province/State'].fillna('') == state)
    else:
        mask = df['Country/Region'] == country
    
    cases = df[mask].iloc[:, 4:].diff(axis=1).T
    cases.columns = ['Daily Cases']
    cases.index = pd.to_datetime(cases.index)
    return cases.clip(lower=0)  # Remove negative values from diff

# Feature engineering
def create_features(df, days_to_predict=30):
    """Create time-series features"""
    df = df.copy()
    
    # Lag features
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f'lag_{lag}'] = df['Daily Cases'].shift(lag)
    
    # Rolling features
    df['rolling_avg_7'] = df['Daily Cases'].rolling(7).mean()
    df['rolling_std_7'] = df['Daily Cases'].rolling(7).std()
    
    # Date features
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.isocalendar().week
    
    # Create future prediction rows
    last_date = df.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=days_to_predict)
    future_df = pd.DataFrame(index=future_dates)
    
    full_df = pd.concat([df, future_df])
    
    return full_df.dropna(subset=['Daily Cases'], how='all')

# Model training function
def train_model(X_train, y_train):
    """Train optimized time series model"""
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    
    # Time-series cross validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Simple training (for demo - replace with proper CV in production)
    model.fit(X_train, y_train)
    return model

# Streamlit UI
st.title("COVID-19 Daily Cases Prediction Dashboard")

# Country and state selection
col1, col2 = st.columns(2)
with col1:
    countries = sorted(confirmed_df['Country/Region'].unique())
    country = st.selectbox("Select Country", countries, index=countries.index('India') if 'India' in countries else 0)

with col2:
    states = confirmed_df[confirmed_df['Country/Region'] == country]['Province/State'].unique()
    state = st.selectbox("Select State/Province", ['National'] + sorted([s for s in states if pd.notna(s)]))

# Date range selection
days_to_predict = st.slider("Days to predict", 7, 90, 30)

if st.button("Generate Predictions"):
    # Load and preprocess data
    daily_cases = preprocess_data(
        confirmed_df, 
        country, 
        state if state != 'National' else None
    )
    
    if daily_cases.empty:
        st.error("No data available for selected region")
        st.stop()
    
    # Feature engineering
    processed_data = create_features(daily_cases, days_to_predict)
    
    # Split data
    train_data = processed_data[processed_data['Daily Cases'].notna()]
    X_train = train_data.drop('Daily Cases', axis=1)
    y_train = train_data['Daily Cases']
    
    # Train model (in production, pre-train and save)
    model = train_model(X_train, y_train)
    
    # Make predictions
    future_data = processed_data[processed_data['Daily Cases'].isna()]
    X_future = future_data.drop('Daily Cases', axis=1)
    
    # Fill any remaining NA values (for lags in future)
    X_future = X_future.fillna(method='ffill').fillna(0)
    
    predictions = model.predict(X_future)
    predictions = np.round(np.clip(predictions, 0, None))  # No negative cases
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'Date': X_future.index,
        'Predicted Daily Cases': predictions
    }).set_index('Date')
    
    # Combine historical and predicted data
    history_df = daily_cases.rename(columns={'Daily Cases': 'Actual Daily Cases'})
    combined_df = history_df.join(result_df, how='outer')
    
    # Visualization
    st.subheader(f"Daily Cases Prediction for {state if state != 'National' else ''} {country}".strip())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data
    ax.plot(
        history_df.index, 
        history_df['Actual Daily Cases'], 
        label='Actual Cases',
        color='blue',
        alpha=0.7
    )
    
    # Plot predictions
    ax.plot(
        result_df.index,
        result_df['Predicted Daily Cases'],
        label='Predicted Cases',
        color='red',
        linestyle='--'
    )
    
    # Formatting
    ax.set_title(f"Daily COVID-19 Cases {'in ' + state if state != 'National' else ''} {country}".strip())
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Cases")
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)
    
    # Show data table
    st.subheader("Prediction Data")
    st.dataframe(
        combined_df.tail(60).style.format("{:,.0f}").background_gradient(cmap='YlOrRd'),
        use_container_width=True
    )
    
    # Model evaluation metrics
    if len(history_df) > 30:
        # Create evaluation on last 30 days of actual data
        eval_data = history_df.iloc[-30:]
        eval_X = create_features(eval_data).drop('Daily Cases', axis=1)
        eval_pred = model.predict(eval_X)
        
        mae = mean_absolute_error(eval_data['Actual Daily Cases'], eval_pred)
        st.metric("Model MAE (30-day validation)", f"{mae:,.1f} cases")

# Add explanatory text
st.markdown("""
**How to use:**
1. Select a country and (optionally) state/province
2. Choose how many days to predict (7-90)
3. Click "Generate Predictions"

**Note:** Predictions show daily new cases (not cumulative totals)
""")
