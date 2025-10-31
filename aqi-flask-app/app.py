from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# ✅ Model path — looks inside the same folder as app.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), "aqi.pkl")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Feature names
FEATURE_NAMES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]

def aqi_category(aqi_value):
    if aqi_value <= 50: return "Good", "#009966"
    if aqi_value <= 100: return "Satisfactory", "#ffde33"
    if aqi_value <= 200: return "Moderate", "#ff9933"
    if aqi_value <= 300: return "Poor", "#cc0033"
    if aqi_value <= 400: return "Very Poor", "#660099"
    return "Severe", "#7e0023"

@app.route("/")
def index():
    return render_template("index.html", features=FEATURE_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        X = np.array([float(data[f]) for f in FEATURE_NAMES]).reshape(1, -1)
        pred = float(model.predict(X)[0])
        category, color = aqi_category(pred)
        return jsonify({"aqi": round(pred, 2), "category": category, "color": color})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
