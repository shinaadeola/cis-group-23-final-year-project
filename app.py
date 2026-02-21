import os
import re
import numpy as np
import joblib
from flask import Flask, request, render_template, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "super_secret_academic_key"  # Needed for flash messages if used

# Load models safely on startup
model_path = os.path.join("models", "isoforest.pkl")
scaler_path = os.path.join("models", "scaler.pkl")

iso_model = None
scaler = None

if os.path.exists(model_path) and os.path.exists(scaler_path):
    iso_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    print("Warning: Models not found in models/ directory!")

def extract_custom_features(text):
    if not isinstance(text, str):
        text = str(text)
    length = len(text)
    html_tags = len(re.findall(r'<[^>]+>', text))
    urls = len(re.findall(r'(http[s]?://|www\.)', text.lower()))
    exclamations = text.count('!')
    dollar_signs = text.count('$')
    uppers = sum(1 for c in text if c.isupper())
    upper_ratio = uppers / length if length > 0 else 0
    text_lower = text.lower()
    urgent_words = text_lower.count('urgent') + text_lower.count('immediate') + text_lower.count('action required')
    account_words = text_lower.count('account') + text_lower.count('suspend') + text_lower.count('verify')
    
    return [length, html_tags, urls, exclamations, dollar_signs, upper_ratio, urgent_words, account_words]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if iso_model is None or scaler is None:
        return "System Error: Machine learning models are missing. Ensure train.py was run.", 500
        
    email_text = request.form.get('emailInput', '')
    
    if not email_text.strip():
        # If empty form was submitted, just send them back to the home page
        return redirect(url_for('index'))
        
    # 1. Extract features
    features = extract_custom_features(email_text)
    
    # 2. Scale features
    features_array = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features_array)
    
    # 3. Predict anomaly (-1 is anomaly, 1 is normal)
    prediction = int(iso_model.predict(scaled_features)[0])
    score = float(iso_model.decision_function(scaled_features)[0])
    
    status = "Threat" if prediction == -1 else "Safe"
    
    feature_breakdown = [
        {"name": "Total Length", "value": features[0]},
        {"name": "Hidden HTML Tags", "value": features[1]},
        {"name": "Clickable Links", "value": features[2]},
        {"name": "Exclamation Marks", "value": features[3]},
        {"name": "Dollar Signs", "value": features[4]},
        {"name": "ALL CAPS Ratio", "value": f"{features[5]:.2f}"},
        {"name": "Urgent Keywords", "value": features[6]},
        {"name": "Account Keywords", "value": features[7]},
    ]
    
    return render_template('result.html', 
                           status=status, 
                           score=score, 
                           features=feature_breakdown,
                           email_snippet=email_text[:200] + ("..." if len(email_text) > 200 else ""))

if __name__ == '__main__':
    # Run locally on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
