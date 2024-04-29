from flask import Flask, render_template, request
import numpy as np
import joblib

# Load your trained machine learning model
model = joblib.load('model.pkl')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        gender = int(request.form['gender'])
        married = int(request.form['married'])
        dependents = int(request.form['dependents'])
        education = int(request.form['education'])
        self_employed = int(request.form['self_employed'])
        applicant_income = float(request.form['applicant_income'])
        coapplicant_income = float(request.form['coapplicant_income'])
        loan_amount = float(request.form['loan_amount'])
        loan_amount_term = float(request.form['loan_amount_term'])
        credit_history = float(request.form['credit_history'])
        property_area = int(request.form['property_area'])

        # Prepare input data for prediction
        input_data = np.array([[gender, married, dependents, education, self_employed,
                                applicant_income, coapplicant_income, loan_amount,
                                loan_amount_term, credit_history, property_area]])

        # Make prediction
        # Probability of getting the loan
        probability = model.predict_proba(input_data)[0][1]

        # Convert probability to percentage
        probability_percent = round(probability * 100, 2)

        return render_template('result.html', probability=probability_percent)


if __name__ == '__main__':
    app.run(debug=True)
