from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import os

# Create a simple model for testing
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create some dummy data for training
X = np.random.rand(100, 13)  # 13 features
y = np.random.randint(0, 2, 100)  # Binary classification

# Train the model
model.fit(X, y)

# Save the model
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist
model_path = os.path.join(model_dir, 'heart_disease_model.pkl')
joblib.dump(model, model_path)

print(f"Model saved to {model_path}") 