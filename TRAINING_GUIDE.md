# Complete LLM Training Guide - UCL COMP0220

## Overview

This is a complete, professional-grade implementation of a Transformer-based Language Model trained from scratch, meeting all coursework requirements.

## What's Included

### Model 1: GPT-2 Style Transformer FROM SCRATCH
- **Architecture**: 12-layer Transformer (124M parameters)
  - 768 hidden dimensions
  - 12 attention heads
  - 256 maximum sequence length
  - Dropout regularization (0.1)

### Datasets (3 Sources - Requirement Met)
1. **Cornell Movie Dialogs** (100,000 samples)
   - Real conversation pairs from movies
   - Auto-downloaded from Cornell website

2. **ELI5 - Explain Like I'm 5** (50,000 samples)
   - AI literacy explanations
   - Attempts to load from Hugging Face, falls back to curated data

3. **Technical Deep Learning Q&A** (30,000 samples)
   - Technical questions and answers about deep learning
   - Curated for AI literacy focus

**Total: 180,000 training samples**

### Features
- ✅ Complete training pipeline with error handling
- ✅ Comprehensive logging (file + console)
- ✅ Checkpointing and resume capability
- ✅ Learning rate scheduling (warmup + cosine annealing)
- ✅ Regularization (Dropout, L2 weight decay, Gradient clipping)
- ✅ Evaluation metrics (Perplexity, BLEU, Hallucination detection)
- ✅ Progress tracking and time estimates
- ✅ GPU utilization

## File Structure

```
DeepLearningCoursework/
├── COMPLETE_LLM_TRAINING.py    # Main training script
├── llm/
│   ├── transformer_model.py    # GPT-2 style Transformer architecture
│   ├── dataset_utils.py        # Dataset loading utilities
│   └── model_evaluator.py      # Evaluation metrics
└── checkpoints/                # Saved models (created during training)
    ├── MODEL1_BEST.pt          # Best model (lowest validation loss)
    ├── MODEL1_FINAL.pt         # Final model after training
    └── vocab.json              # Vocabulary mapping
```

## Training Configuration

### Hyperparameters
- **Epochs**: 30
- **Batch Size**: 32
- **Learning Rate**: 3e-4 (with warmup)
- **Weight Decay**: 0.01 (L2 regularization)
- **Gradient Clipping**: 1.0
- **Optimizer**: AdamW
- **Scheduler**: Warmup (2 epochs) + Cosine Annealing

### Expected Training Time
- **RTX 3090 Ti**: ~6-8 hours
- **RTX 4060**: ~10-12 hours

### Expected Performance
- **Perplexity**: 20-30 (lower is better)
- **Vocabulary**: ~30,000 tokens

## How to Run

### 1. Connect to UCL GPU
```bash
ssh -J muhamaaz@knuckles.cs.ucl.ac.uk muhamaaz@canada-l.cs.ucl.ac.uk
cd /cs/student/projects1/2023/muhamaaz/DeepLearningCoursework
```

### 2. Install Dependencies
```bash
pip3 install --user torch transformers datasets tqdm requests numpy scipy
```

### 3. Run Training
```bash
python3 COMPLETE_LLM_TRAINING.py
```

### 4. Resume from Checkpoint (if interrupted)
Edit `COMPLETE_LLM_TRAINING.py`:
```python
RESUME_FROM = "checkpoints/checkpoint_epoch_15.pt"  # Path to checkpoint
```
Then run again.

## Training Output

### Console Output
- Real-time progress bars
- Epoch summaries (loss, perplexity, learning rate)
- Time estimates
- Evaluation results

### Log Files
- Saved in `logs/training_YYYYMMDD_HHMMSS.log`
- Contains all training metrics and messages

### Checkpoints
- **MODEL1_BEST.pt**: Best model (saved whenever validation loss improves)
- **checkpoint_epoch_N.pt**: Periodic checkpoints (every 5 epochs)
- **MODEL1_FINAL.pt**: Final model after all epochs

## Evaluation Metrics

### During Training
- **Perplexity**: Calculated every 2 epochs on validation set
- **BLEU Score**: Text generation quality
- **Hallucination Detection**: Tracks potential hallucinations in generated text

### Model Files Include
- Model weights
- Optimizer state
- Vocabulary
- Configuration
- Training metrics

## Coursework Requirements Status

✅ **Two LLM models**: 
   - Model 1: Transformer from scratch (this script)
   - Model 2: Mistral 7B fine-tuned (separate script)

✅ **3+ datasets**: Cornell (100K) + ELI5 (50K) + Technical (30K)

✅ **AI literacy focus**: ELI5 + Technical Q&A datasets

✅ **Professional grade**: 
   - 180K samples
   - 30 epochs
   - Proper architecture (12 layers)
   - Comprehensive evaluation

✅ **Regularization**: Dropout, L2 weight decay, Gradient clipping

✅ **FREE resources only**: All datasets and models are free

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` in `TrainingConfig`
- Reduce `MAX_LEN` (e.g., 128 instead of 256)

### Slow Training
- Reduce dataset sizes (CORNELL_SAMPLES, ELI5_SAMPLES, etc.)
- Use fewer epochs for testing

### Checkpoint Resume Not Working
- Ensure checkpoint path is correct
- Check that vocabulary file exists (`checkpoints/vocab.json`)

### Dataset Download Issues
- Cornell dataset: Check internet connection
- ELI5 dataset: Will fall back to curated data if Hugging Face unavailable

## Next Steps

1. **Train Model 1**: Run `COMPLETE_LLM_TRAINING.py`
2. **Train Model 2**: Run `COMPLETE_MISTRAL_FINETUNE.py` (separate script)
3. **Compare Results**: Both models will have evaluation metrics
4. **Prepare Podcast**: Use trained models for AI literacy podcast

## Questions?

Check the code comments for detailed explanations of each component.

