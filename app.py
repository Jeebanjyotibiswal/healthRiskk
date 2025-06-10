from flask import Flask, render_template, request
import pickle
import numpy as np
from werkzeug.utils import secure_filename
import os
import json
import joblib

app = Flask(__name__)

UPLOAD_FOLDER = 'static/reports'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Save it as a joblib model
joblib.dump(model,"model.joblib") 
symptoms = [
    "Chest Pain", "Shortness of Breath", "Irregular Heartbeat", "Fatigue & Weakness",
    "Dizziness", "Swelling (Edema)", "Pain in Neck/Jaw/Shoulder/Back", "Excessive Sweating",
    "Persistent Cough", "Nausea/Vomiting", "High Blood Pressure", "Chest Discomfort (Activity)",
    "Cold Hands/Feet", "Snoring/Sleep Apnea", "Anxiety/Feeling of Doom"
]


risk_factors = {
    "Chest Pain": "Chest pain can indicate reduced blood flow to heart",
    "Shortness of Breath": "Difficulty breathing may signal heart or lung issues",
    "Irregular Heartbeat": "Arrhythmias can increase cardiovascular risk",
    "Age": "Risk increases with age, especially after 45 for men, 55 for women"
}


@app.route("/")
def index():
    return render_template("index.html", 
                         symptoms=symptoms,
                         page_title="Health Risk Predictor | Home")


@app.route("/predict", methods=["POST"])
def predict():
    try:
      
        input_data = []
        symptom_values = {}
        
        for symptom in symptoms:
            value = int(request.form.get(symptom, 0))
            input_data.append(value)
            symptom_values[symptom] = value

        
        age = int(request.form.get("Age", 30))
        input_data.append(age)

        
        final_input = np.array(input_data).reshape(1, -1)

        
        raw_prediction = model.predict(final_input)[0]
        prediction = max(0, min(100, round(raw_prediction, 2)))

        
        top_factors = []
        if age > 45:
            top_factors.append(f"Age ({age}) increased risk by {min(20, age//5)}%")
        
       
        symptom_contributors = [s for s in symptoms if symptom_values.get(s, 0) == 1][:3]
        for i, symptom in enumerate(symptom_contributors):
            top_factors.append(f"{symptom} contributed {15 - i*5}%")
        if prediction > 70:
            risk_level = "high"
            recommendation = "Immediate consultation recommended"
        elif prediction > 30:
            risk_level = "moderate"
            recommendation = "Preventive measures advised"
        else:
            risk_level = "low"
            recommendation = "Maintain healthy lifestyle"

        return render_template("index.html",
                            symptoms=symptoms,
                            prediction=prediction,
                            risk_level=risk_level,
                            recommendation=recommendation,
                            top_factors=top_factors,
                            age=age,
                            page_title=f"Results | {prediction}% Risk")

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return render_template("index.html",
                            symptoms=symptoms,
                            prediction="Error",
                            error_message="Could not process your request. Please try again.")

    except Exception as e:
        app.logger.error(f"Report generation error: {str(e)}")
        return {
            'status': 'error',
            'message': 'Could not generate report'
        }, 500

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
