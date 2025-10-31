from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import gdown  # ✅ for downloading model from Google Drive

app = Flask(__name__)

# === Model file setup ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), "aqi.pkl")

# ✅ Google Drive file ID (replace with your own)
# Example link: https://drive.google.com/file/d/1AbCDefGhIJklMNopQR/view?usp=sharing
# Then file ID = 1AbCDefGhIJklMNopQR
DRIVE_FILE_ID = "1WBRhEifzfGb0FGpvTe0FIBPuiLs8HxBA"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

# === Download model if not present ===
if not os.path.exists(MODEL_PATH):
    print("📥 Model not found locally. Downloading from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    print("✅ Model downloaded successfully!")

# === Load model ===
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# === Feature names ===
FEATURE_NAMES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]

# === AQI category function ===
def aqi_category(aqi_value):
    if aqi_value <= 50: return "Good", "#009966"
    if aqi_value <= 100: return "Satisfactory", "#ffde33"
    if aqi_value <= 200: return "Moderate", "#ff9933"
    if aqi_value <= 300: return "Poor", "#cc0033"
    if aqi_value <= 400: return "Very Poor", "#660099"
    return "Severe", "#7e0023"

# === Routes ===
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

# === Run ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
