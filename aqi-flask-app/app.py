from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import requests

app = Flask(__name__)

# Path to model
MODEL_PATH = "aqi.pkl"

# Optional: Direct link to your model (e.g. from Google Drive, HuggingFace, etc.)
MODEL_URL = "https://example.com/aqi.pkl"  # 🔗 Replace with your real link

def download_model():
    """Download the model file if not present (useful for Render)."""
    if MODEL_URL.startswith("http"):
        print(f"Downloading model from {MODEL_URL} ...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Model downloaded successfully ✅")
        else:
            raise Exception(f"Failed to download model. Status: {response.status_code}")
    else:
        raise FileNotFoundError(f"Model file not found and no valid MODEL_URL provided.")

# Try loading model
if not os.path.exists(MODEL_PATH):
    try:
        download_model()
    except Exception as e:
        raise FileNotFoundError(f"❌ Model file not found and could not be downloaded: {e}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURE_NAMES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"
]

def aqi_category(aqi_value: float):
    a = aqi_value
    if a <= 50: return "Good", "#009966"
    if a <= 100: return "Satisfactory", "#ffde33"
    if a <= 200: return "Moderate", "#ff9933"
    if a <= 300: return "Poor", "#cc0033"
    if a <= 400: return "Very Poor", "#660099"
    return "Severe", "#7e0023"

@app.route("/")
def index():
    return render_template("index.html", features=FEATURE_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        inputs = []
        for fname in FEATURE_NAMES:
            val = data.get(fname)
            if val is None:
                alt = data.get(fname.replace(".", "").replace("_", ""))
                val = alt
            if val is None:
                return jsonify({"error": f"Missing feature: {fname}"}), 400
            try:
                fval = float(val)
            except:
                return jsonify({"error": f"Invalid value for {fname}: {val}"}), 400
            inputs.append(fval)

        X = np.array(inputs).reshape(1, -1)
        pred = model.predict(X)
        pred_value = float(pred[0]) if hasattr(pred, "__len__") else float(pred)

        category, color = aqi_category(pred_value)

        return jsonify({
            "aqi": round(pred_value, 2),
            "category": category,
            "color": color
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
