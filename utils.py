import pandas as pd
import os
import hashlib

# File paths
DATA_DIR = "data"
USERS_FILE = os.path.join(DATA_DIR, "users.csv")
PATIENTS_FILE = os.path.join(DATA_DIR, "patients.csv")

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Initialize Users File (Default Admin)
if not os.path.exists(USERS_FILE):
    pd.DataFrame({
        "username": ["admin"],
        "password": ["admin123"],  # In production, hash this!
        "name": ["Administrator"]
    }).to_csv(USERS_FILE, index=False)

# Initialize Patients File
if not os.path.exists(PATIENTS_FILE):
    pd.DataFrame(columns=[
        "Patient ID", "Name", "Age", "Gender", "Blood Group", "Contact", "City/Village", "Medical History"
    ]).to_csv(PATIENTS_FILE, index=False)

HISTORY_FILE = os.path.join(DATA_DIR, "history.csv")
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=[
        "Patient ID", "Disease", "Risk Score", "Date", "Inputs"
    ]).to_csv(HISTORY_FILE, index=False)

def login_user(username, password):
    """Simple authentication function for Admins."""
    users = pd.read_csv(USERS_FILE)
    # Ensure columns are treated as strings for comparison
    users['username'] = users['username'].astype(str)
    users['password'] = users['password'].astype(str)
    
    user = users[(users['username'] == username) & (users['password'] == password)]
    if not user.empty:
        return user.iloc[0]['name']
    return None

def login_patient(identifier, contact):
    """Authentication for Patients using Name/ID and Contact Number."""
    df = get_patients()
    if df.empty:
        return None
        
    # Convert to string for safe comparison
    df['Patient ID'] = df['Patient ID'].astype(str)
    df['Name'] = df['Name'].astype(str)
    df['Contact'] = df['Contact'].astype(str)
    
    # Check if identifier matches Name OR ID, AND Contact matches
    # Case insensitive search for name
    patient = df[
        ((df['Patient ID'] == identifier) | (df['Name'].str.lower() == identifier.lower())) & 
        (df['Contact'] == contact)
    ]
    
    if not patient.empty:
        return patient.iloc[0].to_dict()
    return None

from datetime import datetime

def save_prediction(patient_id, disease, risk_score, inputs=None):
    """Save a prediction result to history."""
    if not os.path.exists(HISTORY_FILE):
        # Create with Inputs column if not exists
        pd.DataFrame(columns=[
            "Patient ID", "Disease", "Risk Score", "Date", "Inputs"
        ]).to_csv(HISTORY_FILE, index=False)
        
    history = pd.read_csv(HISTORY_FILE)
    
    # Ensure Inputs column exists (for backward compatibility)
    if 'Inputs' not in history.columns:
        history['Inputs'] = ""
        
    new_record = pd.DataFrame([{
        "Patient ID": str(patient_id),
        "Disease": disease,
        "Risk Score": risk_score,
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Inputs": str(inputs) if inputs else ""
    }])
    history = pd.concat([history, new_record], ignore_index=True)
    history.to_csv(HISTORY_FILE, index=False)
    return True

def get_patient_history(patient_id):
    """Retrieve history for a specific patient."""
    if not os.path.exists(HISTORY_FILE):
        return []
        
    df = pd.read_csv(HISTORY_FILE)
    df = df.fillna("") # Handle NaNs
    # Filter by Patient ID
    patient_history = df[df['Patient ID'].astype(str) == str(patient_id)]
    return patient_history.to_dict('records')

def get_patients():
    """Load all patients."""
    if os.path.exists(PATIENTS_FILE):
        df = pd.read_csv(PATIENTS_FILE)
        return df.fillna("") # Handle NaN values to prevent display issues
    return pd.DataFrame()

def add_patient(patient_data):
    """Add a new patient to the CSV."""
    df = get_patients()
    new_patient = pd.DataFrame([patient_data])
    df = pd.concat([df, new_patient], ignore_index=True)
    df.to_csv(PATIENTS_FILE, index=False)
    return True

def get_patient_by_id(patient_id):
    """Retrieve patient details by ID."""
    df = get_patients()
    if df.empty:
        return None
        
    # Ensure robust comparison by converting both to string and stripping whitespace
    # Handle case where ID might be float (e.g. 1800.0) from CSV read
    df['Patient ID'] = df['Patient ID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    search_id = str(patient_id).strip().replace('.0', '')
    
    patient = df[df['Patient ID'] == search_id]
    
    if not patient.empty:
        return patient.iloc[0].to_dict() # Ensure dict return
DOCTOR_FILE = os.path.join(DATA_DIR, "../datasets/doctor_dataset.csv")

def get_doctor_search_options():
    """Get unique cities and specializations for the dropdowns."""
    if not os.path.exists(DOCTOR_FILE):
        return [], []
        
    try:
        df = pd.read_csv(DOCTOR_FILE, encoding='utf-8') # Handle potential encoding issues
    except UnicodeDecodeError:
        df = pd.read_csv(DOCTOR_FILE, encoding='latin1')
        
    cities = sorted(df['Hospital Location'].dropna().unique().tolist())
    specializations = sorted(df['Doctor Specialization'].dropna().unique().tolist())
    return cities, specializations

def search_doctors(city=None, specialization=None):
    """Search doctors by city and specialization."""
    if not os.path.exists(DOCTOR_FILE):
        return []
        
    try:
        df = pd.read_csv(DOCTOR_FILE, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(DOCTOR_FILE, encoding='latin1')
    
    # Filter
    if city and city != 'Select City':
        df = df[df['Hospital Location'] == city]
        
    if specialization and specialization != 'Select Specialization':
        df = df[df['Doctor Specialization'] == specialization]
        
    return df.to_dict('records')
