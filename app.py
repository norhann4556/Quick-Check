from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import os
import csv
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

# Load the model
try:
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'heart_disease_model.pkl')
    logger.info(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Ensure predictions directory exists
predictions_dir = os.path.join(os.path.dirname(__file__), 'predictions')
os.makedirs(predictions_dir, exist_ok=True)

def save_prediction(patient_info, medical_data, prediction_data):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(predictions_dir, f'prediction_{timestamp}.csv')
    
    # Save detailed patient record
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # Basic patient info
        writer.writerow(['Timestamp', 'Patient Name', 'Email'])
        writer.writerow([timestamp, patient_info['name'], patient_info['email']])
        
        # Medical parameters
        writer.writerow([])  # Empty row for separation
        writer.writerow(['Medical Parameters'])
        for key, value in medical_data.items():
            writer.writerow([key, value])
        
        # Prediction results
        writer.writerow([])  # Empty row for separation
        writer.writerow(['Prediction Results'])
        writer.writerow(['Risk Level', prediction_data['result']])
        writer.writerow(['Probability', prediction_data['probability']])
        writer.writerow(['Disease Type', prediction_data['disease_type']])
        writer.writerow(['Severity', prediction_data['severity']])
    
    # Also save to a master CSV for all patients
    master_file = os.path.join(predictions_dir, 'all_predictions.csv')
    file_exists = os.path.isfile(master_file)
    
    with open(master_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header if file doesn't exist
            writer.writerow(['Timestamp', 'Patient Name', 'Email', 'Age', 'Sex', 'Chest Pain Type', 
                            'Resting BP', 'Cholesterol', 'Blood Sugar', 'ECG Results', 'Max Heart Rate',
                            'Exercise Angina', 'ST Depression', 'ST Slope', 'Major Vessels', 'Thalassemia',
                            'Risk Level', 'Probability', 'Disease Type', 'Severity'])
        
        # Write data row
        writer.writerow([
            timestamp,
            patient_info['name'],
            patient_info['email'],
            medical_data['age'],
            medical_data['sex'],
            medical_data['cp'],
            medical_data['trestbps'],
            medical_data['chol'],
            medical_data['fbs'],
            medical_data['restecg'],
            medical_data['thalach'],
            medical_data['exang'],
            medical_data['oldpeak'],
            medical_data['slope'],
            medical_data['ca'],
            medical_data['thal'],
            prediction_data['result'],
            prediction_data['probability'],
            prediction_data['disease_type'],
            prediction_data['severity']
        ])

def get_heart_disease_type(features):
    """
    Determine the specific type of heart disease based on input features.
    Returns a tuple of (disease_type, description, severity)
    """
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = features[0]
    
    # Initialize severity score (0-100)
    severity = 0
    clinical_notes = []
    
    # Analyze features to determine disease type and severity
    if cp == 0:  # Typical angina
        severity += 30
        clinical_notes.append("Typical angina symptoms present")
        if exang == 1:  # Exercise induced angina
            severity += 20
            clinical_notes.append("Exercise-induced symptoms")
            if oldpeak > 2:
                severity += 15
                clinical_notes.append("Significant ST depression during exercise")
                return "Coronary Artery Disease (CAD) with Exercise-Induced Ischemia", "Classic presentation of CAD with exercise-induced chest pain and significant ST depression. This suggests significant coronary artery narrowing.", severity
            return "Coronary Artery Disease (CAD)", "Typical angina with exercise-induced symptoms, suggesting significant coronary artery narrowing.", severity
        return "Stable Angina", "Typical angina symptoms suggesting coronary artery disease. Pain is predictable and occurs with physical exertion.", severity
    
    elif cp == 1:  # Atypical angina
        severity += 25
        clinical_notes.append("Atypical angina symptoms")
        if oldpeak > 2:  # Significant ST depression
            severity += 25
            clinical_notes.append("Significant ST depression at rest")
            if trestbps > 140:
                severity += 10
                clinical_notes.append("Elevated blood pressure")
                return "Unstable Angina with Hypertension", "Atypical chest pain with significant ST depression and elevated blood pressure. This is a high-risk presentation requiring immediate attention.", severity
            return "Unstable Angina", "Atypical chest pain with significant ST depression, suggesting possible acute coronary syndrome.", severity
        return "Variant Angina", "Atypical chest pain pattern, possibly due to coronary artery spasm. Further evaluation recommended.", severity
    
    elif cp == 2:  # Non-anginal pain
        severity += 15
        clinical_notes.append("Non-anginal chest pain")
        if thal == 2:  # Fixed defect
            severity += 20
            clinical_notes.append("Fixed perfusion defect")
            if ca > 0:
                severity += 15
                clinical_notes.append(f"{int(ca)} major vessels affected")
                return "Previous Myocardial Infarction with Multi-vessel Disease", f"Evidence of previous heart attack with {int(ca)} affected vessels. Requires comprehensive cardiac evaluation.", severity
            return "Previous Myocardial Infarction", "Evidence of previous heart attack with fixed perfusion defect. Regular follow-up recommended.", severity
        return "Non-Cardiac Chest Pain", "Chest pain not typical of cardiac origin. Consider other etiologies.", severity
    
    elif cp == 3:  # Asymptomatic
        if ca > 0:  # Blocked vessels
            severity += 35
            clinical_notes.append(f"{int(ca)} major vessels affected")
            if thalach < 100:
                severity += 15
                clinical_notes.append("Reduced exercise capacity")
                return "Silent Ischemia with Reduced Exercise Capacity", f"Significant coronary artery disease ({int(ca)} vessels affected) without typical symptoms. Reduced exercise capacity noted.", severity
            return "Silent Ischemia", f"Significant coronary artery disease ({int(ca)} vessels affected) without typical symptoms. Regular monitoring recommended.", severity
        return "No Significant Heart Disease", "No current evidence of significant heart disease. Continue regular preventive care.", severity
    
    return "Unknown", "Unable to determine specific type. Further evaluation recommended.", severity

def get_risk_factors(features):
    """
    Analyze features to identify specific risk factors
    Returns a list of identified risk factors with clinical significance
    """
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = features[0]
    risk_factors = []
    
    # Age risk
    if age > 65:
        risk_factors.append({
            "factor": "Advanced age (65+)",
            "significance": "High",
            "details": "Age is a non-modifiable risk factor for cardiovascular disease"
        })
    elif age > 45:
        risk_factors.append({
            "factor": "Middle age (45-65)",
            "significance": "Moderate",
            "details": "Increased cardiovascular risk with age"
        })
    
    # Blood pressure risk
    if trestbps > 140:
        risk_factors.append({
            "factor": "Stage 2 Hypertension",
            "significance": "High",
            "details": f"Resting blood pressure of {trestbps} mmHg exceeds normal range"
        })
    elif trestbps > 120:
        risk_factors.append({
            "factor": "Elevated Blood Pressure",
            "significance": "Moderate",
            "details": f"Resting blood pressure of {trestbps} mmHg is above optimal range"
        })
    
    # Cholesterol risk
    if chol > 240:
        risk_factors.append({
            "factor": "High Cholesterol",
            "significance": "High",
            "details": f"Total cholesterol of {chol} mg/dL exceeds recommended levels"
        })
    elif chol > 200:
        risk_factors.append({
            "factor": "Borderline High Cholesterol",
            "significance": "Moderate",
            "details": f"Total cholesterol of {chol} mg/dL is above optimal range"
        })
    
    # Heart rate risk
    if thalach < 100:
        risk_factors.append({
            "factor": "Reduced Exercise Capacity",
            "significance": "High",
            "details": f"Maximum heart rate of {thalach} bpm indicates reduced cardiovascular fitness"
        })
    
    # Exercise induced angina
    if exang == 1:
        risk_factors.append({
            "factor": "Exercise-Induced Angina",
            "significance": "High",
            "details": "Chest pain during physical activity suggests possible coronary artery disease"
        })
    
    # ST depression
    if oldpeak > 2:
        risk_factors.append({
            "factor": "Significant ST Depression",
            "significance": "High",
            "details": f"ST depression of {oldpeak} mm indicates possible myocardial ischemia"
        })
    
    # Number of major vessels
    if ca > 0:
        risk_factors.append({
            "factor": f"Multi-vessel Disease",
            "significance": "High",
            "details": f"{int(ca)} major coronary vessels showing significant narrowing"
        })
    
    # Thalassemia
    if thal == 2:
        risk_factors.append({
            "factor": "Fixed Perfusion Defect",
            "significance": "High",
            "details": "Evidence of previous myocardial damage"
        })
    elif thal == 1:
        risk_factors.append({
            "factor": "Reversible Perfusion Defect",
            "significance": "Moderate",
            "details": "Indicates possible reversible myocardial ischemia"
        })
    
    # Additional clinical factors
    if fbs == 1:
        risk_factors.append({
            "factor": "Elevated Fasting Blood Sugar",
            "significance": "Moderate",
            "details": "May indicate impaired glucose metabolism"
        })
    
    if restecg == 2:
        risk_factors.append({
            "factor": "Abnormal ECG",
            "significance": "High",
            "details": "ST-T wave changes suggesting possible myocardial ischemia"
        })
    
    return risk_factors

@app.route('/')
def welcome():
    # Clear any existing session data when returning to home
    session.clear()
    return render_template('welcome.html')

@app.route('/patient-info', methods=['GET', 'POST'])
def patient_info():
    if request.method == 'POST':
        # Save form data to session
        session['patient_info'] = {
            'name': request.form.get('name'),
            'age': request.form.get('age'),
            'gender': request.form.get('gender'),
            'email': request.form.get('email')
        }
        return redirect(url_for('index'))
    
    # Restore saved data if exists
    saved_data = session.get('patient_info', {})
    return render_template('patient_info.html', saved_data=saved_data)

@app.route('/index', methods=['GET'])
def index():
    # Check if patient info exists in session
    if 'patient_info' not in session:
        return redirect(url_for('patient_info'))
    return render_template('index.html', patient_info=session['patient_info'])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Check if patient info exists in session
    if 'patient_info' not in session:
        logger.warning("No patient info in session, redirecting to patient_info")
        return redirect(url_for('patient_info'))
    
    if request.method == 'POST':
        try:
            logger.info("Processing prediction request")
            logger.info(f"Form data: {request.form}")
            
            # Save medical data to session
            medical_data = {
                'age': int(request.form.get('age')),
                'sex': int(request.form.get('sex')),
                'cp': int(request.form.get('cp')),
                'trestbps': int(request.form.get('trestbps')),
                'chol': int(request.form.get('chol')),
                'fbs': int(request.form.get('fbs')),
                'restecg': int(request.form.get('restecg')),
                'thalach': int(request.form.get('thalach')),
                'exang': int(request.form.get('exang')),
                'oldpeak': float(request.form.get('oldpeak')),
                'slope': int(request.form.get('slope')),
                'ca': int(request.form.get('ca')),
                'thal': int(request.form.get('thal'))
            }
            logger.info(f"Medical data processed: {medical_data}")
            session['medical_data'] = medical_data
            
            # Prepare features for prediction
            features = np.array([[
                medical_data['age'],
                medical_data['sex'],
                medical_data['cp'],
                medical_data['trestbps'],
                medical_data['chol'],
                medical_data['fbs'],
                medical_data['restecg'],
                medical_data['thalach'],
                medical_data['exang'],
                medical_data['oldpeak'],
                medical_data['slope'],
                medical_data['ca'],
                medical_data['thal']
            ]])
            logger.info(f"Features array shape: {features.shape}")
            
            # Get prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            logger.info(f"Prediction: {prediction}, Probability: {probability}")
            
            # Get disease type and risk factors
            disease_type, disease_description, severity = get_heart_disease_type(features)
            risk_factors = get_risk_factors(features)
            logger.info(f"Disease type: {disease_type}, Description: {disease_description}, Severity: {severity}")
            logger.info(f"Risk factors: {risk_factors}")
            
            # Calculate confidence level
            confidence = 0.8 + (probability * 0.2)
            logger.info(f"Confidence level: {confidence}")
            
            # Save prediction data
            prediction_data = {
                'result': 'High Risk' if prediction == 1 else 'Low Risk',
                'probability': probability,
                'confidence': confidence,
                'disease_type': disease_type,
                'disease_description': disease_description,
                'severity': severity,
                'risk_factors': risk_factors
            }
            logger.info(f"Prediction data prepared: {prediction_data}")
            
            # Save to CSV
            save_prediction(session['patient_info'], medical_data, prediction_data)
            logger.info("Prediction saved to CSV")
            
            # Pass all required data to the template
            template_data = {
                'prediction': prediction,
                'probability': probability,
                'confidence': confidence,
                'disease_type': disease_type,
                'disease_description': disease_description,
                'severity': severity,
                'risk_factors': risk_factors,
                'patient_info': session['patient_info']
            }
            logger.info(f"Template data prepared: {template_data}")
            
            return render_template('predict.html', **template_data)
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}", exc_info=True)
            # If there's an error, restore the form with previous data
            return render_template('index.html', 
                                patient_info=session['patient_info'],
                                medical_data=session.get('medical_data', {}),
                                error=str(e))
    
    # For GET requests, restore saved data if exists
    return render_template('index.html', 
                         patient_info=session['patient_info'],
                         medical_data=session.get('medical_data', {}))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
