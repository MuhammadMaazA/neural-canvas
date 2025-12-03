#!/bin/bash
#
# Train Both LLM Models for CNN Explanation
# ==========================================
# Run this script to train both models sequentially
#
# Usage:
#   ./train_both_models.sh          # Train both models
#   ./train_both_models.sh model1   # Train only Model 1 (from scratch)
#   ./train_both_models.sh model2   # Train only Model 2 (fine-tuned)
#

set -e

cd /cs/student/projects1/2023/muhamaaz/neural-canvas

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment"
fi

echo "========================================"
echo "NEURAL CANVAS - LLM TRAINING"
echo "========================================"
echo "Training LLMs to explain CNN art classifications"
echo "========================================"

train_model1() {
    echo ""
    echo "========================================"
    echo "TRAINING MODEL 1: From Scratch"
    echo "========================================"
    echo "This will train a custom transformer (~56M params)"
    echo ""
    
    python llm/scripts/train_model1_cnn_explainer.py
    
    echo ""
    echo "✓ Model 1 training complete!"
    echo "  Checkpoint: /cs/student/projects1/2023/muhamaaz/checkpoints/model1_cnn_explainer/"
}

train_model2() {
    echo ""
    echo "========================================"
    echo "TRAINING MODEL 2: Fine-tuned"
    echo "========================================"
    echo "This will fine-tune DistilGPT-2 (82M params)"
    echo ""
    
    python llm/scripts/train_model2_cnn_explainer.py
    
    echo ""
    echo "✓ Model 2 training complete!"
    echo "  Checkpoint: /cs/student/projects1/2023/muhamaaz/checkpoints/model2_cnn_explainer/"
}

# Parse arguments
case "${1:-both}" in
    model1)
        train_model1
        ;;
    model2)
        train_model2
        ;;
    both)
        train_model1
        train_model2
        ;;
    *)
        echo "Usage: $0 [model1|model2|both]"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "TRAINING COMPLETE"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Run demo: python demo_full_pipeline.py"
echo "  2. Compare models: python llm/scripts/demo_cnn_explainer.py"
echo ""

