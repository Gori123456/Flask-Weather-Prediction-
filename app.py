from flask import Flask, request, jsonify, render_template
import requests
import numpy as np
import joblib
import datetime
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

API_KEY = "4ed1d04b3a2278df1de2d159e7a30666"  # Replace with your OpenWeatherMap API Key

# Load trained ML model and expected feature names
model = joblib.load("temp_model.pkl")
model_features = joblib.load("model_features.pkl")

print("✅ Model loaded successfully:", type(model))
print("📌 Expected features:", model_features)  

@app.route("/")
def home():
    return render_template("main.html")  # Serve main.html

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "city" not in data:
            return jsonify({"error": "City not provided"}), 400

        city = data["city"]

        # Fetch weather data from OpenWeatherMap
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        weather_response = requests.get(weather_url)
        weather_data = weather_response.json()

        if weather_data.get("cod") != 200:
            return jsonify({"error": f"Weather API Error: {weather_data.get('message')}"}), 500

        # Extract weather features
        tmin = weather_data["main"]["temp_min"]
        tmax = weather_data["main"]["temp_max"]
        prcp = weather_data.get("rain", {}).get("1h", 0)  # Default to 0 if no rain data

        # Get current date features
        today = datetime.datetime.today()
        year = today.year
        month = today.month
        day = today.day

        # Create input array matching the trained model's feature order
        input_data = np.array([[tmin, tmax, prcp, year, month, day]])

        # Ensure the input features are in the correct order
        input_df = dict(zip(model_features, input_data[0]))  # Match feature names
        input_array = np.array([list(input_df.values())])  # Convert to array

        # Make prediction
        prediction = model.predict(input_array)[0]

        return jsonify({
            "city": city,
            "tmin": tmin,
            "tmax": tmax,
            "prcp": prcp,
            "year": year,
            "month": month,
            "day": day,
            "prediction": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
