from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import requests

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "aqi.pkl")
MODEL_URL = "https://huggingface.co/Ratnambar2/aqi-model/resolve/df1ee14c4692ff74f0a4ca765433912fd59f73a5/aqi.pkl"

model = None  # ✅ don’t load immediately

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            print("📥 Downloading model from Hugging Face...")
            response = requests.get(MODEL_URL)
            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    f.write(response.content)
                print("✅ Model downloaded successfully!")
            else:
                raise Exception(f"Failed to download model. Status code: {response.status_code}")

        print("🔄 Loading model into memory...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ Model ready!")
    return model


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

        model = get_model()  # ✅ only loads when needed
        pred = float(model.predict(X)[0])
        category, color = aqi_category(pred)

        return jsonify({"aqi": round(pred, 2), "category": category, "color": color})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
