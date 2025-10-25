from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import traceback

app = Flask(__name__)
CORS(app)

# Load the trained model safely using absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "disease_predictor_model.pkl")

try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully with {len(model.classes_)} classes.")
    # Ensure feature names are available
    feature_columns = list(model.feature_names_in_)
except Exception as e:
    print("Error loading model:", str(e))
    model = None
    feature_columns = []

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded properly'}), 500

    try:
        # Expecting JSON input with symptom values
        data = request.get_json() or {}
        input_df = pd.DataFrame([data])

        # Align columns to match model features
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_df)[0]
        return jsonify({'predicted_disease': prediction})

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return jsonify({'error': str(e), 'trace': traceback_str}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

