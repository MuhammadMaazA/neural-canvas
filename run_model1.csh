#!/bin/csh
# Model 1 Training Runner

cd /cs/student/projects1/2023/muhamaaz/code/neural-canvas
source /cs/student/projects1/2023/muhamaaz/venv/bin/activate.csh
setenv HF_HOME /cs/student/projects1/2023/muhamaaz/datasets
setenv TRANSFORMERS_CACHE /cs/student/projects1/2023/muhamaaz/datasets

echo "============================================================"
echo "Starting Model 1 Training"
echo "============================================================"
echo "GPU: `nvidia-smi --query-gpu=name --format=csv,noheader`"
echo "Time: `date`"
echo "Log: /cs/student/projects1/2023/muhamaaz/logs/model1_training.log"
echo "============================================================"

cd llm/scripts
python train_model1.py
