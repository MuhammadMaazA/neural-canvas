#!/bin/csh
#
# Persistent Training Script - Runs even when you close your laptop
# Uses nohup to keep training running in background
#
# Usage:
#   ./train_persistent.csh        # Start training
#   tail -f nohup.out             # Monitor training
#   ps aux | grep python          # Check if running
#   pkill -f train_model1.py      # Stop training

echo "=========================================="
echo "PERSISTENT TRAINING - Model 1"
echo "=========================================="
echo "This will keep training even if you disconnect!"
echo ""

# Set cache directories
setenv HF_HOME /cs/student/projects1/2023/muhamaaz/datasets
setenv HF_DATASETS_CACHE /cs/student/projects1/2023/muhamaaz/datasets
setenv TRANSFORMERS_CACHE /cs/student/projects1/2023/muhamaaz/datasets

# Activate conda environment
echo "Activating conda environment..."
source /cs/student/projects1/2023/muhamaaz/miniconda3/bin/activate neural-canvas

# Navigate to project directory
cd /cs/student/projects1/2023/muhamaaz/code/neural-canvas

# Check if already running
set running = `ps aux | grep train_model1.py | grep -v grep | wc -l`
if ( $running > 0 ) then
    echo "ERROR: Training is already running!"
    echo "Running processes:"
    ps aux | grep train_model1.py | grep -v grep
    echo ""
    echo "To stop: pkill -f train_model1.py"
    exit 1
endif

# Start training with nohup (keeps running after logout)
echo "Starting persistent training with nohup..."
echo "Output will be saved to: nohup.out"
echo ""

nohup python llm/scripts/train_model1.py > /cs/student/projects1/2023/muhamaaz/logs/training_persistent.log 2>&1 &

set pid = $!
echo "Training started! Process ID: $pid"
echo ""
echo "Useful commands:"
echo "  Monitor training:  tail -f /cs/student/projects1/2023/muhamaaz/logs/training_persistent.log"
echo "  Check if running:  ps aux | grep train_model1.py"
echo "  Stop training:     pkill -f train_model1.py"
echo "  View checkpoints:  ls -lh /cs/student/projects1/2023/muhamaaz/checkpoints/model1/"
echo ""
echo "You can now close your laptop - training will continue!"
echo "=========================================="

# Wait a moment to check if it started
sleep 3

set still_running = `ps aux | grep train_model1.py | grep -v grep | wc -l`
if ( $still_running > 0 ) then
    echo "✓ Training is running successfully!"
else
    echo "✗ Training failed to start. Check the log file."
    tail -20 /cs/student/projects1/2023/muhamaaz/logs/training_persistent.log
endif
