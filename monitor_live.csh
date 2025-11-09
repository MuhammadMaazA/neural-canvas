#!/bin/csh
#
# LIVE TRAINING MONITOR
# Shows training progress in real-time with auto-refresh
#

echo "=========================================="
echo "LIVE TRAINING MONITOR"
echo "=========================================="
echo "Press Ctrl+C to exit monitoring"
echo ""

while (1)
    clear
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║           NEURAL CANVAS - LIVE TRAINING MONITOR               ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Check if training is running
    echo "📊 TRAINING STATUS:"
    echo "─────────────────────────────────────────────────────────────────"
    set running = `ps aux | grep "train_model1.py" | grep -v grep | wc -l`
    if ( $running > 0 ) then
        echo "✓ Status: RUNNING"
        ps aux | grep "train_model1.py" | grep -v grep | head -1 | awk '{printf "  CPU: %s%%  Memory: %s%%  Time: %s\n", $3, $4, $10}'
    else
        echo "✗ Status: NOT RUNNING"
    endif
    echo ""
    
    # GPU Status
    echo "🎮 GPU STATUS:"
    echo "─────────────────────────────────────────────────────────────────"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | awk '{printf "  GPU: %s%%  Memory: %sMB/%sMB  Temp: %s°C\n", $1, $2, $3, $4}'
    echo ""
    
    # Latest training output
    echo "📝 LATEST TRAINING OUTPUT (last 15 lines):"
    echo "─────────────────────────────────────────────────────────────────"
    if ( -f /cs/student/projects1/2023/muhamaaz/logs/training_restart.log ) then
        tail -15 /cs/student/projects1/2023/muhamaaz/logs/training_restart.log
    else if ( -f /cs/student/projects1/2023/muhamaaz/logs/training_persistent.log ) then
        tail -15 /cs/student/projects1/2023/muhamaaz/logs/training_persistent.log
    else
        echo "  No log file found"
    endif
    echo ""
    echo "─────────────────────────────────────────────────────────────────"
    echo "Auto-refresh every 5 seconds... (Ctrl+C to exit)"
    
    sleep 5
end
