from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

model = joblib.load("prognosis_model.pkl")
features = joblib.load("features.pkl")

app = Flask(__name__)

# ⭐ THIS IS THE IMPORTANT PART
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

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
