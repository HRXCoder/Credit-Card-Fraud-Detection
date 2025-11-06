from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        time = float(request.form['time'])
        amount = float(request.form['amount'])
        
        # Create feature array with only Time and Amount
        features = np.array([[time, amount]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Calculate confidence score
        confidence = max(prediction_proba) * 100
        
        result = {
            'prediction': 'Fraudulent' if prediction == 1 else 'Legitimate',
            'confidence': f'{confidence:.2f}%',
            'is_fraud': bool(prediction),
            'message': 'ALERT: Fraudulent transaction detected!' if prediction == 1 else 'Transaction appears legitimate.',
            'time': time,
            'amount': amount
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)