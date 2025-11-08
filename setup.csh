#!/bin/csh
# Setup script for Neural Canvas LLM training

echo "============================================================"
echo "Neural Canvas LLM Setup"
echo "============================================================"

# Set HuggingFace cache to projects folder
setenv HF_HOME /cs/student/projects1/2023/muhamaaz/datasets
setenv HF_DATASETS_CACHE /cs/student/projects1/2023/muhamaaz/datasets

echo "HuggingFace cache: $HF_HOME"

# Create directories
mkdir -p /cs/student/projects1/2023/muhamaaz/datasets
mkdir -p /cs/student/projects1/2023/muhamaaz/checkpoints
mkdir -p /cs/student/projects1/2023/muhamaaz/logs

echo "Directories created"

# Check Python
python --version

# Check CUDA
if ( $?CUDA_VISIBLE_DEVICES ) then
    echo "GPU: Available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Warning: No GPU detected"
endif

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate venv: source /cs/student/projects1/2023/muhamaaz/venv/bin/activate.csh"
echo "2. Train Model 1: cd llm/scripts && python train_model1.py"
echo "3. Train Model 2: cd llm/scripts && python train_model2.py"
echo "============================================================"
