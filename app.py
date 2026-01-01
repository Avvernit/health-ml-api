from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

model = joblib.load("prognosis_model.pkl")
features = joblib.load("features.pkl")


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

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

