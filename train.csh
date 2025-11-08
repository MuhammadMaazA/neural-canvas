#!/bin/csh
# Quick training script for UCL GPU

echo "=== Neural Canvas LLM Training ==="
echo ""

# Setup environment
source setup.csh

# Check GPU
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Activate virtual environment
if (! -d /cs/student/projects1/2023/muhamaaz/venv) then
    echo "Creating virtual environment..."
    python3 -m venv /cs/student/projects1/2023/muhamaaz/venv
endif

source /cs/student/projects1/2023/muhamaaz/venv/bin/activate.csh

# Install dependencies if needed
if (! -f /cs/student/projects1/2023/muhamaaz/venv/.installed) then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    touch /cs/student/projects1/2023/muhamaaz/venv/.installed
endif

# Choose model
echo "Which model to train?"
echo "1) Model 1 (Modern Transformer, ~10 hours)"
echo "2) Model 2 (Llama 2 LoRA, ~4 hours)"
echo -n "Choice [1/2]: "
set choice = $<

if ("$choice" == "1") then
    echo ""
    echo "Starting Model 1 training..."
    echo "This will take ~10 hours. Monitor with: tail -f /cs/student/projects1/2023/muhamaaz/logs/*.log"
    echo ""
    python llm/scripts/train_model1.py
else if ("$choice" == "2") then
    echo ""
    echo "Starting Model 2 training..."
    echo "This will take ~4 hours. Monitor with: tail -f /cs/student/projects1/2023/muhamaaz/logs/*.log"
    echo ""
    python llm/scripts/train_model2.py
else
    echo "Invalid choice"
    exit 1
endif

echo ""
echo "Training complete! Check checkpoints in /cs/student/projects1/2023/muhamaaz/checkpoints/"
