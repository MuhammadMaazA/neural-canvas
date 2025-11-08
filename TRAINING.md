# Training Guide

## Storage Setup (50GB Projects Folder)

Your projects folder: `/cs/student/projects1/2023/muhamaaz` (50GB)

### Recommended Layout

```
/cs/student/projects1/2023/muhamaaz/
├── code/neural-canvas/          # Your code (~100MB)
├── datasets/                    # Cached datasets (~5-10GB)
├── checkpoints/                 # Model checkpoints (~5-10GB)
├── logs/                        # Training logs (~100MB)
└── venv/                        # Python virtual environment (~2GB)
```

## Quick Start

### 1. Setup Virtual Environment

```bash
cd /cs/student/projects1/2023/muhamaaz
python3 -m venv venv
source venv/bin/activate.csh  # for csh
pip install --upgrade pip
cd code/neural-canvas
pip install -r requirements.txt
```

### 2. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

### 3. Train Model 1 (From Scratch)

```bash
cd llm/scripts
python train_model1.py
```

**Expected:**
- Time: 8-12 hours on A100
- Checkpoints: saved to `../../checkpoints/`
- Logs: saved to `../../logs/`

### 4. Train Model 2 (Llama 2 Fine-tuning)

```bash
python train_model2.py
```

**Expected:**
- Time: 4-6 hours on A100
- Memory: ~16GB GPU with 4-bit quantization
- Checkpoints: saved to `../../checkpoints/model2_llama/`

## Storage Management

### Dataset Caching

Datasets will auto-cache to `~/.cache/huggingface/`

To use projects folder instead:

```bash
export HF_HOME=/cs/student/projects1/2023/muhamaaz/datasets
export HF_DATASETS_CACHE=/cs/student/projects1/2023/muhamaaz/datasets
```

Add to your `.cshrc`:

```csh
setenv HF_HOME /cs/student/projects1/2023/muhamaaz/datasets
setenv HF_DATASETS_CACHE /cs/student/projects1/2023/muhamaaz/datasets
```

### Checkpoint Management

Checkpoints are saved to:
- Model 1: `checkpoints/model1_best.pt` (~800MB)
- Model 2: `checkpoints/model2_llama/` (~50MB LoRA adapters)

Delete intermediate checkpoints to save space:

```bash
cd /cs/student/projects1/2023/muhamaaz/code/neural-canvas
rm -f checkpoints/checkpoint_epoch_*.pt  # Keep only best model
```

## Monitoring Training

### Check GPU Usage

```bash
watch -n 1 nvidia-smi
```

### Monitor Logs

```bash
tail -f logs/training_*.log
```

### Check Disk Usage

```bash
du -sh /cs/student/projects1/2023/muhamaaz/*
```

## Troubleshooting

### Out of Storage

```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/

# Clear old checkpoints
rm -f checkpoints/checkpoint_epoch_*.pt

# Clear pip cache
pip cache purge
```

### Out of GPU Memory

Edit config in training scripts:
- Reduce `BATCH_SIZE` from 16 to 8 or 4
- Reduce `MAX_LEN` from 512 to 256

## Expected Storage Usage

- Datasets (cached): 5-10GB
- Model 1 checkpoints: 1-2GB
- Model 2 checkpoints: 50-100MB (LoRA only)
- Virtual environment: 2GB
- Logs: 100MB

**Total: ~10-15GB** (well within 50GB limit)
