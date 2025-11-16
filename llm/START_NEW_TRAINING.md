# How to Start Your IMPROVED Model 1 Training

## ✅ What's Been Fixed

1. **Dataset**: Art-only (NO AI/conversation confusion) - ~280K diverse samples
2. **Quality**: Natural text (NO repetitive templates)
3. **Training**: 50 epochs with better hyperparameters
4. **Monitoring**: Generation quality tested each epoch
5. **Model**: Larger 50M parameters (was 35M)

---

## 🚀 Quick Start

### Step 1: Clean Up Old Training

```bash
# Optional: Delete old bad model to free space
rm -rf /cs/student/projects1/2023/muhamaaz/checkpoints/model1_custom

# New model will be saved to:
# /cs/student/projects1/2023/muhamaaz/checkpoints/model1_improved
```

### Step 2: Start Training

```bash
cd /cs/student/projects1/2023/muhamaaz/neural-canvas/llm/scripts

# Activate virtual environment
source /cs/student/projects1/2023/muhamaaz/neural-canvas/venv/bin/activate

# Start training (will run in background)
nohup python train_improved_model1.py > training_output.log 2>&1 &

# Get the process ID
echo $! > training.pid
```

### Step 3: Monitor Training

```bash
# Watch training progress
tail -f training_output.log

# Or check specific parts
grep "EPOCH.*RESULTS" training_output.log

# See generation tests
grep -A 5 "GENERATION QUALITY TEST" training_output.log
```

### Step 4: Check Training Plots

Training curves are saved automatically:
```bash
# View the plot (updated each epoch)
eog /cs/student/projects1/2023/muhamaaz/checkpoints/model1_improved/training_curves.png
```

---

## ⏱️ Expected Timeline

- **Loading data**: ~5-10 minutes (280K samples)
- **Per epoch**: ~20-30 minutes (depends on GPU)
- **Total training**: 12-24 hours (will auto-stop with early stopping)

---

## 📊 What to Look For

### Good Signs:
- ✅ Perplexity **decreasing** (going down to ~5-8)
- ✅ Val loss **decreasing**
- ✅ Generation tests producing **relevant art content**
- ✅ Train-val gap **small** (< 0.5)

### Bad Signs:
- ❌ Perplexity **increasing** or stuck high (>15)
- ❌ Val loss **increasing** (overfitting)
- ❌ Generation tests still producing **gibberish**
- ❌ Train-val gap **large** (> 1.0) - overfitting

---

## 🎯 Testing After Training

Once training completes, test your model:

```bash
cd /cs/student/projects1/2023/muhamaaz/neural-canvas/llm/scripts

# Test the improved model
python chat_with_models.py --model model1 \
  --model1-path /cs/student/projects1/2023/muhamaaz/checkpoints/model1_improved/best_model.pt \
  --test
```

You should see outputs like:
```
Q: What is Impressionism?
A: Impressionism is an art movement from 1860s France characterized by
visible brushstrokes, emphasis on capturing light, and depicting ordinary
subjects. Key artists include Claude Monet, Pierre-Auguste Renoir...
```

Instead of "the cochlea" 😂

---

## 🛑 Stopping Training Early

If you need to stop:

```bash
# Find the process
cat training.pid

# Kill it
kill $(cat training.pid)

# Training will resume from last checkpoint if you restart
```

---

## 📁 Important Files

```
/cs/student/projects1/2023/muhamaaz/checkpoints/model1_improved/
├── best_model.pt          # Best model (use this!)
├── latest.pt              # Latest checkpoint (for resuming)
├── checkpoint_epoch_*.pt  # Periodic checkpoints
└── training_curves.png    # Training progress plot

/cs/student/projects1/2023/muhamaaz/logs/
└── model1_improved_*.log  # Detailed training logs
```

---

## 💡 Pro Tips

1. **Training will auto-resume** if you restart - it loads from `latest.pt`
2. **Early stopping** will kick in if no improvement for 10 epochs
3. **Best model is saved** whenever validation loss improves
4. **Generation tests** let you see quality improving in real-time

---

## 🎓 For Your Coursework

This improved Model 1 demonstrates:
- ✅ **Training from scratch** (as required)
- ✅ **Challenges**: Needed focused, quality data (not random mixing)
- ✅ **Improvements**: Better dataset design, longer training
- ✅ **Results**: Should now generate coherent art knowledge

Compare with Model 2 (pretrained fine-tuning) to show the difference!

---

## 🆘 Troubleshooting

**Problem**: Out of memory error
**Solution**: Reduce batch size in `train_improved_model1.py`:
```python
BATCH_SIZE = 8  # Was 16
```

**Problem**: Dataset loading too slow
**Solution**: Reduce dataset sizes in `train_improved_model1.py`:
```python
WIKIART_NATURAL = 50000  # Was 100000
```

**Problem**: Training taking too long
**Solution**: Reduce epochs:
```python
NUM_EPOCHS = 30  # Was 50
```

---

**Ready to start?** Run Step 2 above and let it train!

The new model will be WAY better than the current "cochlea" nonsense. 🎨
