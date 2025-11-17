#!/bin/bash
# Continuous training progress monitor
# Shows real-time updates of training progress

LOG_FILE="/cs/student/projects1/2023/muhamaaz/neural-canvas/llm/scripts/training.log"

echo "=================================="
echo "TRAINING PROGRESS MONITOR"
echo "=================================="
echo "Press Ctrl+C to stop monitoring"
echo ""

# Function to get latest epoch info
show_progress() {
    echo "📊 Latest Completed Epochs:"
    echo "-----------------------------------"
    grep "Epoch [0-9]* - Train:" "$LOG_FILE" | tail -5
    
    echo ""
    echo "🔄 Current Training Step:"
    echo "-----------------------------------"
    tail -3 "$LOG_FILE" | grep -E "(Training:|Step)"
    
    echo ""
    echo "💾 Saved Checkpoints:"
    echo "-----------------------------------"
    ls -lth /cs/student/projects1/2023/muhamaaz/checkpoints/model1_custom_v2_fixed_wikiart/*.pt 2>/dev/null | head -4
    
    echo ""
    echo "⏰ Last update: $(date)"
    echo "=================================="
}

# Show initial status
show_progress

echo ""
echo "Monitoring training every 30 seconds..."
echo "(Press Ctrl+C to stop)"
echo ""

# Update every 30 seconds
while true; do
    sleep 30
    clear
    show_progress
done
