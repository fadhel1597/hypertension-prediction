import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from keras.models import load_model
import argparse

app = Flask(__name__)

# Define the mean and std values for standardization (replace with your actual values)
mean = np.array([5.37250e+01, 4.75000e-01, 7.50000e-01, 1.24575e+02, 2.44000e+02, 1.00000e-01, 6.75000e-01, 1.49150e+02, 3.00000e-30, 1.27000e+00, 1.30000e+00, 7.00000e-01, 2.25000e+00])
std = np.array([16.34717493,  0.50573633,  1.00638984, 16.20206935, 66.04970933, 0.30382181,  0.57233216, 23.42642602,  0.46409548,  1.21891082, 0.72324057,  1.1140133 ,  0.66986413])

# Parse command line arguments to determine which model to use
parser = argparse.ArgumentParser(description="Choose model type for prediction")
parser.add_argument("--model-type", type=str, choices=["keras", "scikit-learn"], required=True)
args = parser.parse_args()

if args.model_type == "keras":
    # Load the trained Keras model
    keras_model = load_model("weights/ANN.h5")

@app.route("/hypertension-prediction", methods=["POST"])
def hypertension_prediction():
    try:
        data = request.get_json()
        data = pd.DataFrame([data])

        # Standardize the new data point using the same scaling parameters as used for training
        X_pred_scaled = (data - mean) / std

        if args.model_type == "keras":
            # Make predictions using the Keras model
            y_pred_prob = keras_model.predict(X_pred_scaled)
            threshold = 0.4
            y_pred = (y_pred_prob[:, 0] >= threshold).astype(int)  # Adjust the column index based on your model's output
        else:
            # Load and use the scikit-learn based model
            file = open("weights/SVM.pkl", "rb")
            trained_model = joblib.load(file)

            # Make predictions using the scikit-learn model
            y_pred_prob = trained_model.predict_proba(X_pred_scaled)
            threshold = 0.4
            y_pred = (y_pred_prob[:, 1] >= threshold).astype(int)  # Adjust the column index based on your model's output

        result = {
            "predicted_class": int(y_pred[0]),  # Ensure that the predicted class is of type int
            "probability_of_positive_class": float(y_pred_prob[0][0]) if args.model_type == "keras" else float(y_pred_prob[0][1])  # Ensure that the probability is of type float
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)