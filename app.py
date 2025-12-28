from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Load model + feature list
model = joblib.load("prognosis_model.pkl")
features = joblib.load("features.pkl")

app = Flask(__name__)
CORS(app)   # <--- THIS enables CORS for Wix


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json   # read json from request

    # Put inputs in EXACT SAME ORDER as training
    row = [[data[f] for f in features]]
    df = pd.DataFrame(row, columns=features)

    pred = model.predict(df)[0]
    prob = model.predict_proba(df).max()

    return jsonify({
        "prognosis": pred,
        "confidence": float(prob)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
