#!/bin/csh
#
# QUICK STATUS CHECK
# Run this anytime to see if training is running
#

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║          NEURAL CANVAS - TRAINING STATUS                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check if running
set running = `ps aux | grep train_model1.py | grep -v grep | wc -l`

if ( $running > 0 ) then
    echo "✅ STATUS: TRAINING IS RUNNING"
    echo ""
    ps aux | grep train_model1.py | grep -v grep | awk '{printf "   PID: %s\n   CPU: %s%%\n   Memory: %s%%\n   Runtime: %s\n", $2, $3, $4, $10}'
    
    # Check parent process
    set pid = `ps aux | grep train_model1.py | grep -v grep | awk '{print $2}' | head -1`
    set ppid = `ps -o ppid= -p $pid | tr -d ' '`
    
    echo ""
    if ( "$ppid" == "1" ) then
        echo "✅ SURVIVES DISCONNECT: YES (parent=1, using nohup)"
    else
        echo "⚠️  SURVIVES DISCONNECT: NO (parent=$ppid)"
        echo "   Run: ./start_persistent_training.csh"
    endif
else
    echo "❌ STATUS: NOT RUNNING"
    echo ""
    echo "To start training:"
    echo "   ./start_persistent_training.csh"
endif

echo ""
echo "───────────────────────────────────────────────────────────"
echo "USEFUL COMMANDS:"
echo "───────────────────────────────────────────────────────────"
echo "  Monitor live:    tail -f /cs/student/projects1/2023/muhamaaz/logs/training_persistent.log"
echo "  Stop training:   pkill -f train_model1.py"
echo "  View curves:     ls -lh /cs/student/projects1/2023/muhamaaz/checkpoints/model1/*.png"
echo "  Check GPU:       nvidia-smi"
echo ""
echo "YOU CAN CLOSE YOUR LAPTOP - TRAINING WILL CONTINUE!"
echo ""
