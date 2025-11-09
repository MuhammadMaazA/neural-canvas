#!/bin/csh
#
# Screen-based Persistent Training - RECOMMENDED for UCL GPUs
# Keeps training alive even when you disconnect SSH
#
# Usage:
#   ./train_screen.csh              # Start new training session
#   screen -r training              # Reconnect to training session
#   screen -ls                      # List all screen sessions
#   
# Inside screen:
#   Ctrl+A then D                   # Detach (training keeps running)
#   exit                            # Stop training and close screen

echo "=========================================="
echo "SCREEN-BASED PERSISTENT TRAINING"
echo "=========================================="
echo "Recommended for UCL GPU training"
echo ""

# Set cache directories
setenv HF_HOME /cs/student/projects1/2023/muhamaaz/datasets
setenv HF_DATASETS_CACHE /cs/student/projects1/2023/muhamaaz/datasets
setenv TRANSFORMERS_CACHE /cs/student/projects1/2023/muhamaaz/datasets

# Check if screen session already exists
set existing = `screen -ls | grep training | wc -l`
if ( $existing > 0 ) then
    echo "Training session already exists!"
    echo ""
    screen -ls
    echo ""
    echo "Options:"
    echo "  1. Reconnect:  screen -r training"
    echo "  2. Kill old:   screen -X -S training quit"
    echo "  3. Then rerun this script"
    exit 1
endif

# Create a wrapper script for screen
cat > /tmp/train_wrapper_$$.csh << 'EOF'
#!/bin/csh

# Set cache directories
setenv HF_HOME /cs/student/projects1/2023/muhamaaz/datasets
setenv HF_DATASETS_CACHE /cs/student/projects1/2023/muhamaaz/datasets
setenv TRANSFORMERS_CACHE /cs/student/projects1/2023/muhamaaz/datasets

# Activate conda
source /cs/student/projects1/2023/muhamaaz/miniconda3/bin/activate neural-canvas

# Navigate to project
cd /cs/student/projects1/2023/muhamaaz/code/neural-canvas

# Show info
echo "=========================================="
echo "TRAINING STARTED IN SCREEN SESSION"
echo "=========================================="
echo "Session name: training"
echo "To detach: Press Ctrl+A then D"
echo "To reconnect later: screen -r training"
echo "=========================================="
echo ""

# Run training
python llm/scripts/train_model1.py

# Keep shell open after training finishes
echo ""
echo "Training completed or stopped."
echo "Press Enter to close this screen session..."
set response = $<
EOF

chmod +x /tmp/train_wrapper_$$.csh

# Start screen session with the training
echo "Starting screen session named 'training'..."
echo ""
screen -dmS training /tmp/train_wrapper_$$.csh

sleep 2

echo "✓ Screen session created!"
echo ""
echo "Useful commands:"
echo "  Reconnect to training:  screen -r training"
echo "  List all sessions:      screen -ls"
echo "  Kill training:          screen -X -S training quit"
echo ""
echo "Inside the screen session:"
echo "  Detach (keep running):  Ctrl+A then D"
echo "  Stop training:          Ctrl+C then type 'exit'"
echo ""
echo "Now reconnecting to the session..."
echo "Press Ctrl+A then D to detach and keep training running!"
echo ""

sleep 2
screen -r training

# Cleanup
rm -f /tmp/train_wrapper_$$.csh
