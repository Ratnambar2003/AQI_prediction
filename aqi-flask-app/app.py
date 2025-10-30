from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model
MODEL_PATH = "aqi.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Place your aqi.pkl in project root.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# If your model requires a scaler or encoder that was saved separately,
# load them here the same way. For example:
# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)
# If not, we assume model accepts raw numeric features in the given order.

FEATURE_NAMES = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]


def aqi_category(aqi_value: float):
    """Return (category_name, color_hex) for display."""
    a = aqi_value
    if a <= 50:
        return "Good", "#009966"
    if a <= 100:
        return "Satisfactory", "#ffde33"
    if a <= 200:
        return "Moderate", "#ff9933"
    if a <= 300:
        return "Poor", "#cc0033"
    if a <= 400:
        return "Very Poor", "#660099"
    return "Severe", "#7e0023"

@app.route("/")
def index():
    return render_template("index.html", features=FEATURE_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        # collect features in the expected order
        inputs = []
        for fname in FEATURE_NAMES:
            # allow both camel/pretty names or exact keys
            val = data.get(fname)
            if val is None:
                # try alternate keys like PM2_5 or PM2.5 if user sent that
                alt = data.get(fname.replace(".", "").replace("_", ""))
                val = alt
            if val is None:
                return jsonify({"error": f"Missing feature: {fname}"}), 400
            # convert to float
            try:
                fval = float(val)
            except:
                return jsonify({"error": f"Invalid value for {fname}: {val}"}), 400
            inputs.append(fval)

        X = np.array(inputs).reshape(1, -1)

        # If your model needs scaling, apply scaler here before predict.
        # X = scaler.transform(X)

        pred = model.predict(X)
        # ensure scalar
        if hasattr(pred, "__len__"):
            pred_value = float(pred[0])
        else:
            pred_value = float(pred)

        category, color = aqi_category(pred_value)

        return jsonify({
            "aqi": round(pred_value, 2),
            "category": category,
            "color": color
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
