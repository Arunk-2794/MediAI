# Run this script in PowerShell to setup and launch MediAI

Write-Host "Starting MediAI Setup..." -ForegroundColor Green

# Ensure we are in the correct directory
Set-Location -Path $PSScriptRoot

# 1. Install Dependencies
Write-Host "Installing Requirements..." -ForegroundColor Cyan
pip install -r requirements.txt

# 2. Generate Data (if not exists)
if (-not (Test-Path "datasets\heart.csv")) {
    Write-Host "Generating Synthetic Data..." -ForegroundColor Cyan
    python generate_data.py
}
else {
    Write-Host "Data already exists. Skipping generation." -ForegroundColor Yellow
}

# 3. Train Models (if not exists)
if (-not (Test-Path "models\heart_model.pkl")) {
    Write-Host "Training Models..." -ForegroundColor Cyan
    python train_models.py
}
else {
    Write-Host "Models already trained. Skipping training." -ForegroundColor Yellow
}

# 4. Launch App
Write-Host "Launching Flask App..." -ForegroundColor Green
python app.py
