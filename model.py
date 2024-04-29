import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
loan_data = pd.read_csv("data.csv")

# Handling Null Values
loan_data["Gender"] = loan_data["Gender"].fillna(loan_data["Gender"].mode()[0])
loan_data["Married"] = loan_data["Married"].fillna(
    loan_data["Married"].mode()[0])
loan_data["Dependents"] = loan_data["Dependents"].fillna(
    loan_data["Dependents"].mode()[0])
loan_data["Self_Employed"] = loan_data["Self_Employed"].fillna(
    loan_data["Self_Employed"].mode()[0])
loan_data["LoanAmount"] = loan_data["LoanAmount"].fillna(
    loan_data["LoanAmount"].median())
loan_data["Loan_Amount_Term"] = loan_data["Loan_Amount_Term"].fillna(
    loan_data["Loan_Amount_Term"].median())
loan_data["Credit_History"] = loan_data["Credit_History"].fillna(
    loan_data["Credit_History"].median())

# Outliers Detection and Handling
loan_data = loan_data[loan_data["ApplicantIncome"] < 25000]
loan_data = loan_data[loan_data["CoapplicantIncome"] < 12000]
loan_data = loan_data[loan_data["LoanAmount"] < 400]

# Data Preparation
loan_data = loan_data.drop(["Loan_ID"], axis=1)
loan_data["Gender"] = loan_data["Gender"].replace(("Male", "Female"), (1, 0))
loan_data["Married"] = loan_data["Married"].replace(("Yes", "No"), (1, 0))
loan_data["Self_Employed"] = loan_data["Self_Employed"].replace(
    ("Yes", "No"), (1, 0))
loan_data["Education"] = loan_data["Education"].replace(
    ("Graduate", "Not Graduate"), (1, 0))
loan_data["Loan_Status"] = loan_data["Loan_Status"].replace(("Y", "N"), (1, 0))
loan_data["Property_Area"] = loan_data["Property_Area"].replace(
    ("Urban", "Semiurban", "Rural"), (1, 1, 0))
loan_data["Dependents"] = loan_data["Dependents"].replace(
    ("0", "1", "2", "3+"), (0, 1, 1, 1))

# Train Test Split
y = loan_data["Loan_Status"]
X = loan_data.drop(["Loan_Status"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model Building
model = LogisticRegression()
model.fit(x_train, y_train)

# Pickling the model
joblib.dump(model, 'model.pkl')
