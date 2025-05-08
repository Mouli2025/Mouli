
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load and preprocess data
def load_and_preprocess_data():
    df = pd.read_csv("customer.csv")
    df.columns = df.columns.str.strip()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop("customerID", axis=1, inplace=True)

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1})

    df = pd.get_dummies(df, drop_first=True)

    scaler = StandardScaler()
    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
        df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return X, y, scaler, X.columns.tolist()

# Train the model once when the app starts
X, y, scaler, feature_columns = load_and_preprocess_data()
model = XGBClassifier(eval_metric='logloss')
model.fit(X, y)

# Set up Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        input_df = pd.DataFrame([input_data])

        # Match the training features
        input_df = pd.get_dummies(input_df)
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_columns]

        # Scale numerical features
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
            input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
        )

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0, 1]
        return jsonify({"churn_prediction": int(prediction), "probability": float(proba)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
