from flask import Flask, request, jsonify, render_template
import joblib
import re

app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocess function
def preprocess_text(text):
    # Clean and preprocess the text
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    # Vectorize and transform the input text
    vectorized_text = vectorizer.transform([text])
    return vectorized_text

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get("text")
    processed_text = preprocess_text(text)
    
    # Make a prediction
    probability = model.predict_proba(processed_text)[0][1]  # Probability for class 1 (abusive)
    threshold = 0.55
    result = "abusive" if probability >= threshold else "not abusive"
    
    return render_template(
        'index.html',
        prediction_text=f"Text: {text}\nProbability: {probability:.2f}\nClassification: {result}"
    )

if __name__ == "__main__":
    app.run(debug=True)
