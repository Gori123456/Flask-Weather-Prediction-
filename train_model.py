import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import joblib

# Load dataset
data = pd.read_csv("bombay.csv")

# Check for missing values before handling
print("Missing values before handling:\n", data.isnull().sum())

# Fill missing values using median strategy
imputer = SimpleImputer(strategy="median")
data[["tavg", "tmin", "tmax", "prcp"]] = imputer.fit_transform(data[["tavg", "tmin", "tmax", "prcp"]])

# Drop any remaining NaN values
data.dropna(inplace=True)

# ✅ Convert 'datetime' column (Specify format explicitly)
data["datetime"] = pd.to_datetime(data["datetime"], format="%d-%m-%Y", errors="coerce")

# Extract year, month, and day from the datetime column
data["year"] = data["datetime"].dt.year
data["month"] = data["datetime"].dt.month
data["day"] = data["datetime"].dt.day

# Drop rows where datetime conversion failed
data.dropna(subset=["datetime"], inplace=True)

# Define features and target variable
X = data.drop(columns=["tavg", "datetime"])  # Features: tmin, tmax, prcp, year, month, day
y = data["tavg"]  # Target: Average temperature

# Split dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "temp_model.pkl")

# Save feature names
joblib.dump(X.columns.tolist(), "model_features.pkl")

print("✅ Model trained and saved successfully!")
