#!/bin/csh
#
# Start training that SURVIVES when you close your laptop
# Uses nohup - training continues even after SSH disconnect
#

echo "=========================================="
echo "PERSISTENT TRAINING - SURVIVES DISCONNECT"
echo "=========================================="
echo ""

# Check if already running
set running = `ps aux | grep "train_model1.py" | grep -v grep | wc -l`
if ( $running > 0 ) then
    echo "✓ Training is ALREADY running!"
    echo ""
    ps aux | grep "train_model1.py" | grep -v grep
    echo ""
    echo "Options:"
    echo "  1. Monitor live:        tail -f /cs/student/projects1/2023/muhamaaz/logs/training_restart.log"
    echo "  2. Kill and restart:    pkill -f train_model1.py"
    echo ""
    exit 0
endif

# Kill any tail processes first
pkill -f "tail -f" >& /dev/null

# Setup environment
cd /cs/student/projects1/2023/muhamaaz/code/neural-canvas
source /cs/student/projects1/2023/muhamaaz/venv/bin/activate.csh

# Start with nohup (immune to hangup signal)
echo "Starting training with nohup..."
echo "Training will continue even if you:"
echo "  - Close your laptop"
echo "  - Disconnect SSH"
echo "  - Log out"
echo ""

nohup python llm/scripts/train_model1.py >& /cs/student/projects1/2023/muhamaaz/logs/training_persistent.log &

set pid = $!
echo "✓ Training started! Process ID: $pid"
echo ""
echo "=========================================="
echo "MONITORING COMMANDS:"
echo "=========================================="
echo ""
echo "1. Live monitoring (updates in real-time):"
echo "   tail -f /cs/student/projects1/2023/muhamaaz/logs/training_persistent.log"
echo ""
echo "2. Check last 50 lines:"
echo "   tail -50 /cs/student/projects1/2023/muhamaaz/logs/training_persistent.log"
echo ""
echo "3. Check if still running:"
echo "   ps aux | grep train_model1.py"
echo ""
echo "4. Stop training:"
echo "   pkill -f train_model1.py"
echo ""
echo "5. View checkpoints:"
echo "   ls -lht /cs/student/projects1/2023/muhamaaz/checkpoints/model1/"
echo ""
echo "=========================================="
echo "YOU CAN NOW CLOSE YOUR LAPTOP SAFELY!"
echo "=========================================="
