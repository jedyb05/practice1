from flask import Flask, render_template, request
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__)

# Assuming 'model' is your trained machine learning model
# Replace 'model' with the actual variable name of your trained model
model = ...

# Save the trained model using joblib
joblib.dump(model, 'model.pkl')

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract input values from the form
        features = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]

        # Make predictions with the loaded model
        prediction = model.predict([features])[0]

        # Display the result page with the prediction
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
