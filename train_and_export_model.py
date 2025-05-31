import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris  # Replace this with your real data
import cloudpickle

# Load or define your training data
X, y = load_iris(return_X_y=True)  # Replace with real (X, y) if available

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Save using cloudpickle for better compatibility
with open("random_forest_model.pkl", "wb") as f:
    cloudpickle.dump(model, f)
