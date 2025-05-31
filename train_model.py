# train_model.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris  # Replace with your actual X/y
import joblib

# Load dummy data for example purposes
X, y = load_iris(return_X_y=True)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Save model with joblib in the *same environment*
joblib.dump(model, "random_forest_model.pkl")

print("Model trained and saved successfully.")
