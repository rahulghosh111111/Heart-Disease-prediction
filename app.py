from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form inputs
        features = [float(request.form[key]) for key in request.form.keys()]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    except Exception as e:
        result = f"Error: {str(e)}"
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
