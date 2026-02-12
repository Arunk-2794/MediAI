import pandas as pd
import numpy as np
import random

def generate_comprehensive_dataset(n=5000):
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    # Diseases to simulate
    conditions = [
        'Healthy', 'Diabetes', 'Hypertension', 'HeartDisease', 
        'StrokeRisk', 'KidneyDisease', 'LiverDisease', 'Asthma'
    ]
    
    for _ in range(n):
        # Base Demographics
        age = np.random.randint(18, 90)
        gender = np.random.choice(['Male', 'Female'])
        
        # Initial Random Vitals (Healthy Baseline)
        bmi = np.random.uniform(18.5, 24.9)
        bp_sys = np.random.randint(90, 119)
        bp_dia = np.random.randint(60, 79)
        glucose = np.random.randint(70, 99)
        chol = np.random.randint(125, 199)
        
        # Lifestyle Factors
        smoking = 'Never'
        alcohol = 'None'
        activity = 'Moderate'
        diet = 'Good'
        sleep = np.random.uniform(6, 9)
        history = 'None'
        
        # Decide Target first, then force features to match (Synthetic Generation Strategy)
        # This ensures strong correlations for the model to learn
        
        # Weighted prob for target
        target = np.random.choice(conditions, p=[0.35, 0.1, 0.1, 0.1, 0.05, 0.1, 0.1, 0.1])
        
        if target == 'Healthy':
            # Keep healthy baseline, maybe slight variations
            if np.random.random() < 0.2: activity = 'High'
            if np.random.random() < 0.1: alcohol = 'Moderate'
            
        elif target == 'Diabetes':
            glucose = np.random.randint(130, 250) # Distinctly high
            bmi = np.random.uniform(28, 40) # Overweight
            history = np.random.choice(['Diabetes', 'None'], p=[0.7, 0.3])
            diet = 'Poor'
            activity = 'Low'
            
        elif target == 'Hypertension':
            bp_sys = np.random.randint(140, 180) # Distinctly high
            bp_dia = np.random.randint(90, 110)
            age = max(age, 40)
            diet = np.random.choice(['Poor', 'Good'], p=[0.7, 0.3]) # Salt?
            history = np.random.choice(['Hypertension', 'None'], p=[0.6, 0.4])

        elif target == 'HeartDisease':
            age = max(age, 50)
            chol = np.random.randint(240, 350)
            bp_sys = np.random.randint(130, 160)
            smoking = np.random.choice(['Current', 'Former'], p=[0.7, 0.3])
            diet = 'Poor'
            history = np.random.choice(['HeartDisease', 'None'], p=[0.6, 0.4])
            
        elif target == 'StrokeRisk':
            age = max(age, 60)
            bp_sys = np.random.randint(150, 200)
            bmi = np.random.uniform(30, 45)
            smoking = 'Current'
            history = np.random.choice(['Stroke', 'None'], p=[0.5, 0.5])
            
        elif target == 'KidneyDisease':
            age = max(age, 45)
            bp_sys = np.random.randint(135, 170)
            glucose = np.random.randint(110, 180) # Slight intersection with diabetes
            history = 'KidneyDisease' # Strong genetic link for synthetic signal
            
        elif target == 'LiverDisease':
            alcohol = 'High' # Strong signal
            age = max(age, 35)
            bmi = np.random.uniform(25, 35)
            history = np.random.choice(['LiverDisease', 'None'], p=[0.4, 0.6])
            
        elif target == 'Asthma':
            # Can be young
            history = 'Asthma' # Strong signal
            smoking = np.random.choice(['Never', 'Former', 'Current'], p=[0.6, 0.2, 0.2])
            # Vitals usually normal unless attack, but model needs static features
            # Maybe slight random noise in others
            
        # Add random noise to non-critical features to prevent overfitting
        if np.random.random() < 0.1: smoking = np.random.choice(['Never', 'Former', 'Current'])
        if np.random.random() < 0.1: sleep = np.random.uniform(4, 10)

        row = [
            age, gender, round(bmi, 1), int(bp_sys), int(bp_dia), int(glucose), int(chol),
            smoking, alcohol, activity, diet, round(sleep, 1), history,
            target
        ]
        data.append(row)
        
    columns = [
        'Age', 'Gender', 'BMI', 'BloodPressure_Systolic', 'BloodPressure_Diastolic',
        'Glucose_Fasting_mg_dL', 'Cholesterol_Total_mg_dL', 
        'Smoking', 'AlcoholIntake', 'PhysicalActivity', 'DietQuality', 'SleepHours',
        'FamilyHistory', 'TargetLabel'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Remove duplicates
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    final_len = len(df)
    
    if initial_len > final_len:
        print(f"Removed {initial_len - final_len} duplicate records.")
    
    df.to_csv("datasets/disease_dataset.csv", index=False)
    print(f"Refined dataset generated with {final_len} unique records.")
    print(df['TargetLabel'].value_counts())

if __name__ == "__main__":
    generate_comprehensive_dataset()
