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

# ✅ ADD THIS HEALTH ENDPOINT
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_count': len(feature_columns) if feature_columns else 0
    })

# ✅ ADD THIS HOME ENDPOINT (optional)
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'name': 'Disease Predictor API',
        'version': '1.0.0',
        'status': 'running',
        'model_loaded': model is not None,
        'features': feature_columns[:10] if feature_columns else []
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight CORS request
    if request.method == 'OPTIONS':
        return '', 200
    
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
        
        # Get confidence score
        try:
            probabilities = model.predict_proba(input_df)[0]
            confidence = max(probabilities)
        except:
            confidence = 1.0
        
        return jsonify({
            'success': True,
            'predicted_disease': prediction,
            'confidence': float(confidence),
            'message': 'Prediction successful'
        })

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return jsonify({
            'success': False,
            'error': str(e),
            'trace': traceback_str
        }), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
