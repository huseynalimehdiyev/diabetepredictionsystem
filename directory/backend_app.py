from flask import Flask, render_template, request
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)
model = joblib.load("diabetes_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form


    height = float(data['height'])
    weight = float(data['weight'])
    bmi = round(weight / (height ** 2), 2)

    input_data = pd.DataFrame([[
        int(data['pregnancies']), float(data['glucose']), float(data['blood_pressure']),
        float(data['skin_thickness']), float(data['insulin']), bmi,
        float(data['dpf']), int(data['age'])
    ]], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])

    prediction = model.predict(input_data)[0]
    probability = 100 - model.predict_proba(input_data)[0][prediction] * 100

    message = "⚠There is a risk of diabetes." if prediction == 1 else "✅ No risk of diabetes"

    return render_template("result.html",
        message=message,
        probability=f"{probability:.1f}",
        date=datetime.now().strftime("%d.%m.%Y %H:%M:%S"),


        pregnancies=data['pregnancies'],
        glucose=data['glucose'],
        blood_pressure=data['blood_pressure'],
        skin_thickness=data['skin_thickness'],
        insulin=data['insulin'],
        height=height,
        weight=weight,
        bmi=bmi,
        dpf=data['dpf'],
        age=data['age']
    )

if __name__ == "__main__":
    app.run(debug=True)
