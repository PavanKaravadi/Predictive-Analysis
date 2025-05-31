import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris  # Replace with your actual data
import joblib

# Load example data (replace with real X/y)
X, y = load_iris(return_X_y=True)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Save model using joblib
joblib.dump(model, "random_forest_model.pkl")

print("✅ Model trained and saved as 'random_forest_model.pkl'")
