#!/bin/bash
# Script to start training with proper virtual environment activation

cd /cs/student/projects1/2023/muhamaaz/neural-canvas
source venv/bin/activate
cd llm/scripts

# Start training in background
nohup python train_custom_model.py > training.log 2>&1 &
echo $! > training.pid
echo "Training started with PID: $(cat training.pid)"
echo "To monitor progress, run: tail -f /cs/student/projects1/2023/muhamaaz/neural-canvas/llm/scripts/training.log"
