#!/bin/csh
# Quick start training script for Model 1

echo "========================================"
echo "Starting Model 1 Training"
echo "========================================"

# Setup environment
setenv HF_HOME /cs/student/projects1/2023/muhamaaz/datasets
setenv HF_DATASETS_CACHE /cs/student/projects1/2023/muhamaaz/datasets
setenv TRANSFORMERS_CACHE /cs/student/projects1/2023/muhamaaz/datasets

# Activate virtual environment
source /cs/student/projects1/2023/muhamaaz/venv/bin/activate.csh

# Run training
cd /cs/student/projects1/2023/muhamaaz/code/neural-canvas
python llm/scripts/train_model1.py

echo "========================================"
echo "Training complete or stopped"
echo "========================================"
