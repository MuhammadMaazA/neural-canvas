#!/bin/csh
#
# Monitor Training Progress
# Shows current training status, GPU usage, and latest metrics
#

echo "=========================================="
echo "TRAINING MONITOR"
echo "=========================================="
echo ""

# Check if training is running
echo "1. Training Process Status:"
echo "----------------------------"
set running = `ps aux | grep train_model1.py | grep -v grep`
if ( "$running" != "" ) then
    echo "✓ Training is RUNNING"
    echo "$running"
else
    echo "✗ Training is NOT running"
endif
echo ""

# Check screen sessions
echo "2. Screen Sessions:"
echo "----------------------------"
screen -ls
echo ""

# Show GPU usage
echo "3. GPU Usage:"
echo "----------------------------"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo ""

# Show latest log entries
echo "4. Latest Training Logs (last 30 lines):"
echo "----------------------------"
if ( -f /cs/student/projects1/2023/muhamaaz/logs/training_persistent.log ) then
    tail -30 /cs/student/projects1/2023/muhamaaz/logs/training_persistent.log
else
    set latest_log = `ls -t /cs/student/projects1/2023/muhamaaz/logs/training_*.log 2>/dev/null | head -1`
    if ( "$latest_log" != "" ) then
        tail -30 "$latest_log"
    else
        echo "No log files found"
    endif
endif
echo ""

# Show checkpoints
echo "5. Saved Checkpoints:"
echo "----------------------------"
ls -lht /cs/student/projects1/2023/muhamaaz/checkpoints/model1/*.pt 2>/dev/null | head -10
echo ""

# Show training curves
echo "6. Training Curves:"
echo "----------------------------"
ls -lh /cs/student/projects1/2023/muhamaaz/checkpoints/model1/*.png 2>/dev/null
echo ""

echo "=========================================="
echo "Useful Commands:"
echo "  Watch logs live:    tail -f /cs/student/projects1/2023/muhamaaz/logs/training_persistent.log"
echo "  Reconnect screen:   screen -r training"
echo "  Stop training:      pkill -f train_model1.py"
echo "  View curves:        Open PNG files in /cs/student/projects1/2023/muhamaaz/checkpoints/model1/"
echo "=========================================="
