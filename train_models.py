import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Create models directory if not exists
if not os.path.exists("models"):
    os.makedirs("models")

def train_unified_model():
    print("Training Unified Disease Model...")
    try:
        df = pd.read_csv("datasets/disease_dataset.csv")
    except FileNotFoundError:
        print("Dataset not found. Please run generate_comprehensive_data.py first.")
        return

    # Features and Target
    # Added: AlcoholIntake, PhysicalActivity, DietQuality, SleepHours
    X = df[['Age', 'Gender', 'BMI', 'BloodPressure_Systolic', 'BloodPressure_Diastolic', 
            'Glucose_Fasting_mg_dL', 'Cholesterol_Total_mg_dL', 'Smoking', 
            'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours',
            'FamilyHistory']]
    y = df['TargetLabel']
    
    # Encoders
    encoders = {}
    # New categorical features to encode
    categorical_cols = ['Gender', 'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'FamilyHistory']
    
    for col in categorical_cols:
        le = LabelEncoder()
        X.loc[:, col] = le.fit_transform(X[col])
        encoders[col] = le
        print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Unified Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    print("Classification Report:\n", classification_report(y_test, model.predict(X_test)))
    
    # Save artifacts
    joblib.dump(model, "models/unified_model.pkl")
    joblib.dump(scaler, "models/unified_scaler.pkl")
    joblib.dump(encoders, "models/unified_encoders.pkl")
    print("Unified Model and Encoders Saved.\n")

if __name__ == "__main__":
    train_unified_model()
