from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)


model_ada = joblib.load('./static/ada_model.pkl')
model_cat = joblib.load('./static/cat_model.pkl')
model_forest = joblib.load('./static/forest_model.pkl')


FEATURE_COLUMNS = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12",
    "V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23",
    "V24","V25","V26","V27","V28","Amount"
]

def build_feature_vector(form):
    values = []

    for field in FEATURE_COLUMNS:
        raw = form.get(field)

        try:
            values.append(float(raw))
        except:
            values.append(0.0)   # fallback an toàn

    return np.array(values).reshape(1, -1)


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    probability = None
    selected_model = "rf"  

    if request.method == "POST":

        selected_model = request.form.get("model", "rf")

        if selected_model == "rf":
            model = model_forest
        elif selected_model == "ada":
            model = model_ada
        else:
            model = model_cat

        X = build_feature_vector(request.form)

        pred = model.predict(X)[0]

        try:
            proba = model.predict_proba(X)[0][1]
        except:
            proba = None

        prediction = "Fraud (Gian lận)" if pred == 1 else "Legitimate (Hợp lệ)"
        if proba is not None:
            probability = round(proba * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        selected_model=selected_model
    )

if __name__ == "__main__":
    app.run(debug=True)
