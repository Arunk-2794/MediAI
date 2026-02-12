from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import numpy as np
import joblib
import os
from utils import login_user, login_patient, get_patients, add_patient as add_patient_data, get_patient_by_id, save_prediction, get_patient_history, get_doctor_search_options, search_doctors

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_hackathon'  # Change this for production


# --- Load Unified Model ---
unified_model = None
unified_scaler = None
unified_encoders = None

def load_models():
    global unified_model, unified_scaler, unified_encoders
    path_model = "models/unified_model.pkl"
    path_scaler = "models/unified_scaler.pkl"
    path_encoders = "models/unified_encoders.pkl"
    
    if os.path.exists(path_model) and os.path.exists(path_scaler) and os.path.exists(path_encoders):
        unified_model = joblib.load(path_model)
        unified_scaler = joblib.load(path_scaler)
        unified_encoders = joblib.load(path_encoders)
        print("Unified Model Loaded Successfully.")
    else:
        print("Unified Model not found. Please train models first.")

load_models()

# --- Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        usertype = request.form.get('usertype', 'admin') # Default to admin if not specified
        
        if usertype == 'patient':
            # Patient Login
            patient = login_patient(username, password) # Identifier, Contact
            if patient:
                session['logged_in'] = True
                session['usertype'] = 'patient'
                session['username'] = patient['Name']
                session['patient_id'] = patient['Patient ID']
                flash(f'Welcome back, {patient["Name"]}!', 'success')
                return redirect(url_for('patient_dashboard'))
            else:
                 flash('Invalid Patient ID or Contact Number.', 'danger')
        else:
            # Admin Login
            name = login_user(username, password)
            if name:
                session['logged_in'] = True
                session['usertype'] = 'admin'
                session['username'] = name
                flash('Admin Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid Admin credentials.', 'danger')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Generate Random Patient ID
        import random
        patients = get_patients()
        existing_ids = patients['Patient ID'].astype(str).values if not patients.empty else []
        
        while True:
            new_id = str(random.randint(1200, 1900))
            if new_id not in existing_ids:
                break
        
        patient_data = {
            "Patient ID": new_id,
            "Name": request.form['name'],
            "Age": request.form['age'],
            "Gender": request.form['gender'],
            "Blood Group": request.form.get('blood_group', ''),
            "Contact": request.form.get('contact', ''),
            "City/Village": request.form.get('city_village', ''),
            "Medical History": ""
        }
        
        add_patient_data(patient_data)
        flash(f'Account created successfully! Your Patient ID is {new_id}. Please login with this ID and your Contact Number.', 'success')
        return redirect(url_for('login'))
            
    return render_template('register.html')

@app.route('/patient_dashboard')
def patient_dashboard():
    if not session.get('logged_in') or session.get('usertype') != 'patient':
        return redirect(url_for('login'))
        
    patient_id = session.get('patient_id')
    patient = get_patient_by_id(patient_id)
    history = get_patient_history(patient_id)
    
    if patient is not None:
        # patient is already a dict from get_patient_by_id
        print("DEBUG: Patient Data for Dashboard:", patient) # Debug print
    else:
        # If patient ID from session is invalid/not found
        session.clear()
        flash('Session expired or invalid. Please login again.', 'warning')
        return redirect(url_for('login'))
        
    return render_template('patient_dashboard.html', patient=patient, history=history)



@app.route('/patient_records/<patient_id>')
def view_patient_records(patient_id):
    if not session.get('logged_in') or session.get('usertype') != 'admin':
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('login'))
        
    patient = get_patient_by_id(patient_id)
    history = get_patient_history(patient_id)
    
    if patient is not None:
        # patient is already a dict
        print("DEBUG: Admin View - Patient Data:", patient)
    else:
        flash('Patient not found.', 'danger')
        print("DEBUG: Admin View - Patient not found")
        return redirect(url_for('dashboard'))
        
    return render_template('patient_dashboard.html', patient=patient, history=history, view_only=True)


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        flash('Please login to access the dashboard.', 'warning')
        return redirect(url_for('login'))
        
    if session.get('usertype') == 'patient':
        return redirect(url_for('patient_dashboard'))
        
    patients_df = get_patients()
    patients = patients_df.to_dict('records') # Convert DataFrame to list of dicts for Jinja
    
    total = len(patients)
    male = len([p for p in patients if p.get('Gender') == 'Male'])
    female = len([p for p in patients if p.get('Gender') == 'Female'])
    
    return render_template('dashboard.html', patients=patients, total_patients=total, male_patients=male, female_patients=female)

@app.route('/predict')
def predict_page():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    patient_id = request.args.get('patient_id', session.get('patient_id')) # Auto-fill self if patient
    
    patient = None
    if patient_id:
        p_data = get_patient_by_id(patient_id)
        if p_data is not None:
            patient = p_data
            
    return render_template('predict.html', patient=patient)

@app.route('/result', methods=['POST'])
def predict_result():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    if unified_model is None:
        flash('Unified Model not loaded. Please train models.', 'danger')
        return redirect(url_for('dashboard'))
        
    try:
        # 1. Collect Input Data
        age = float(request.form['age'])
        gender = request.form['gender']
        bmi = float(request.form['bmi'])
        bp_sys = float(request.form['bp_sys'])
        bp_dia = float(request.form['bp_dia'])
        glucose = float(request.form['glucose'])
        chol = float(request.form['chol'])
        smoking = request.form['smoking']
        alcohol = request.form['alcohol']
        activity = request.form['activity']
        diet = request.form['diet']
        sleep = float(request.form['sleep'])
        family_history = request.form['family_history']
        
        # 2. Encode Categorical Data
        def safe_transform(name, val):
            if name not in unified_encoders:
                return 0
            le = unified_encoders[name]
            if val in le.classes_:
                return le.transform([val])[0]
            # Fallback for unknown classes? Assign most common or 0
            return 0 
            
        gender_enc = safe_transform('Gender', gender)
        smoking_enc = safe_transform('Smoking', smoking)
        alcohol_enc = safe_transform('AlcoholIntake', alcohol)
        activity_enc = safe_transform('PhysicalActivity', activity)
        diet_enc = safe_transform('DietQuality', diet)
        history_enc = safe_transform('FamilyHistory', family_history)
        
        # 3. Create Feature Array
        # Order MUST match training: 
        # Age, Gender, BMI, BP_Sys, BP_Dia, Glucose, Cholesterol, Smoking, Alcohol, Activity, Diet, Sleep, History
        features = [[
            age, gender_enc, bmi, bp_sys, bp_dia, glucose, chol, 
            smoking_enc, alcohol_enc, activity_enc, diet_enc, sleep, 
            history_enc
        ]]
        
        # 4. Scale
        features_scaled = unified_scaler.transform(features)
        
        # 5. Predict
        prediction = unified_model.predict(features_scaled)[0]
        probabilities = unified_model.predict_proba(features_scaled)[0]
        max_prob = max(probabilities)
        
        # 6. Save History
        if session.get('usertype') == 'patient':
            input_details = f"Age:{age}, BMI:{bmi}, BP:{bp_sys}/{bp_dia}, Gluc:{glucose}, Chol:{chol}, Smoke:{smoking}, Alc:{alcohol}, Act:{activity}, Diet:{diet}, Sleep:{sleep}"
            save_prediction(session.get('patient_id'), prediction, round(max_prob * 100, 2), inputs=input_details)
            
        return render_template('result.html', prediction=prediction, probability=max_prob, disease="Unified Analysis")
        
    except Exception as e:
        flash(f'Error processing prediction: {str(e)}', 'danger')
        print(f"Error: {e}") # Debug print
        return redirect(url_for('predict_page'))



@app.route('/find_doctor', methods=['GET', 'POST'])
def find_doctor():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    cities, specializations = get_doctor_search_options()
    
    selected_city = ""
    selected_spec = ""
    doctors = []
    searched = False
    
    if request.method == 'POST':
        searched = True
        selected_city = request.form.get('city')
        selected_spec = request.form.get('specialization')
        doctors = search_doctors(selected_city, selected_spec)
        
    return render_template('find_doctor.html', 
                           cities=cities, 
                           specializations=specializations,
                           doctors=doctors,
                           searched=searched,
                           selected_city=selected_city,
                           selected_spec=selected_spec)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
