#!/bin/bash
# Train with nohup - keeps running even if you disconnect
# Usage: bash train_nohup.sh

echo "Starting training with nohup..."
echo "Training will continue even if you disconnect!"
echo ""

# Create logs directory if it doesn't exist
mkdir -p /cs/student/projects1/2023/muhamaaz/logs

# Start training in background with nohup
nohup python -u llm/scripts/train_art_ai_model.py > /cs/student/projects1/2023/muhamaaz/logs/nohup_training.log 2>&1 &

# Get the process ID
PID=$!

echo "✓ Training started in background"
echo "Process ID: $PID"
echo ""
echo "To monitor training:"
echo "  tail -f /cs/student/projects1/2023/muhamaaz/logs/nohup_training.log"
echo ""
echo "To check if still running:"
echo "  ps -p $PID"
echo ""
echo "To kill training:"
echo "  kill $PID"
echo ""
echo "You can now disconnect safely!"
