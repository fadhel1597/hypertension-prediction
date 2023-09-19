from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

@app.route("/hypertension-predicition", methods=["POST"])
def hypertension_predicition():

    data = request.get_json()
    data = pd.DataFrame(data)

    if model.endswith('.pkl'):

        file = open(model,"rb")
        trained_model = joblib.load(file)

        y_pred_prob = trained_model.predict_proba(data.reshape(1,-1))
        y_pred = (y_pred_prob >= 0.4).astype(int)

        for pred in y_pred:
            if pred[0] == 1:
                class_name = 'Normal'
            else:
                class_name = 'Hypertension'
        
        condition = {
            'condition' : class_name
        }
    elif model.endswith('.h5'):
        pass

    return jsonify(condition)

if __name__ == "__main__":
    model = "weights/SVM.pkl"
    app.run(host="0.0.0.0", port=5001, debug=True)