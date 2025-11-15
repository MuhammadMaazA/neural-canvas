# Model Comparison - Your Coursework Setup

## Your Training Setup

### Model 1: Train from Scratch (Current)
**Script:** `llm/scripts/train_art_ai_model.py`

**Architecture:**
- Transformer from scratch (Modern: RoPE, RMSNorm, GQA, SwiGLU)
- Parameters: **~285M** (calculated below)
- Dimensions: 1024
- Layers: 12
- Attention heads: 16
- KV heads: 4 (Grouped Query Attention)

**Parameter Calculation:**
```
Embedding: 32000 (vocab) × 1024 (dim) = 32.8M
Each layer: ~4 × 1024² = 4.2M per layer
12 layers: 12 × 4.2M = 50.4M
FFN (SwiGLU): 12 × 8 × 1024² = 100M
Total: ~285M parameters
```

**Training:**
- From scratch on ~310K art + AI literacy samples
- Time: 3-5 hours on RTX 3090 Ti
- Expected perplexity: 15-20

---

### Model 2: Pretrained (For Comparison)
**Script:** `llm/scripts/train_model2_pretrained.py` (NEW!)

**Options (pick one):**

#### Option A: GPT-2 Large (Recommended)
- Parameters: **774M** (2.7× larger than yours!)
- Pros: Good quality, manageable size
- Training time: 2-3 hours fine-tuning
- Already trained on massive data

#### Option B: GPT-2 XL (Best)
- Parameters: **1.5B** (5.3× larger!)
- Pros: Best GPT-2 model
- Training time: 3-4 hours fine-tuning
- Highest quality

#### Option C: Phi-2 (Microsoft)
- Parameters: **2.7B** (9.5× larger!)
- Pros: Very efficient, modern
- Training time: 4-5 hours fine-tuning
- Might be too big for 3090 Ti

**Recommendation: Use GPT-2 XL (1.5B params)**

---

## How to Run

### 1. Model 1 (From Scratch) - WITH NOHUP

```bash
# This keeps running even if you disconnect!
bash train_nohup.sh
```

**Monitor:**
```bash
tail -f /cs/student/projects1/2023/muhamaaz/logs/nohup_training.log
```

**Check if running:**
```bash
ps aux | grep train_art_ai_model
```

### 2. Model 2 (Pretrained) - WITH NOHUP

```bash
# Edit the model choice first:
# Open llm/scripts/train_model2_pretrained.py
# Line 38-40: Choose MODEL_NAME

# Then run with nohup:
nohup python -u llm/scripts/train_model2_pretrained.py > logs/model2_nohup.log 2>&1 &
```

**Monitor:**
```bash
tail -f logs/model2_nohup.log
```

---

## Parameter Comparison

| Model | Parameters | Size | Training Time |
|-------|-----------|------|---------------|
| Your Model 1 (scratch) | 285M | 1.1GB | 3-5 hours |
| GPT-2 Base | 124M | 500MB | ❌ Too small |
| GPT-2 Medium | 355M | 1.4GB | 2-3 hours |
| **GPT-2 Large** ⭐ | **774M** | **3GB** | **2-3 hours** |
| **GPT-2 XL** ⭐⭐ | **1.5B** | **6GB** | **3-4 hours** |
| Phi-2 | 2.7B | 10GB | 4-5 hours |

**Recommendation: GPT-2 XL (1.5B) for best comparison**

---

## Coursework: What to Compare

### For Your Podcast:

1. **Model Size:**
   - Model 1: 285M params
   - Model 2: 1.5B params (5× larger!)

2. **Training:**
   - Model 1: Trained from scratch (slower to learn)
   - Model 2: Fine-tuned (already knows language)

3. **Quality:**
   - Model 1: Expected perplexity ~15-20
   - Model 2: Expected perplexity ~10-15 (better!)

4. **Art Knowledge:**
   - Model 1: Only knows what you taught it
   - Model 2: Already has world knowledge + your art data

5. **Cost:**
   - Model 1: More epochs, more energy
   - Model 2: Few epochs, less energy per epoch (but bigger model)

6. **Sustainability:**
   - Model 1: Training from scratch uses more compute
   - Model 2: Fine-tuning is more efficient
   - But Model 2 inference uses more GPU memory

---

## Testing Both Models

```python
# Test Model 1
checkpoint = torch.load("checkpoints/art_ai_model/art_ai_best.pt")
model1 = ModernTransformer(**checkpoint['config'])
model1.load_state_dict(checkpoint['model_state_dict'])

# Test Model 2
from transformers import AutoModelForCausalLM, AutoTokenizer
model2 = AutoModelForCausalLM.from_pretrained("checkpoints/model2_pretrained/final")
tokenizer2 = AutoTokenizer.from_pretrained("checkpoints/model2_pretrained/final")

# Compare outputs
prompt = "What is Impressionism?"
output1 = generate(model1, prompt)
output2 = generate(model2, prompt)

# Which is better?
```

---

## Which Model to Use?

**For your coursework, you MUST train BOTH:**

### Model 1 (Required):
✓ Train from scratch with 3+ datasets
✓ Show techniques: dropout, label smoothing, etc.
✓ Benchmark evaluation
✓ User testing

### Model 2 (Required):
✓ Use pretrained model (GPT-2 XL recommended)
✓ Optional but gets extra marks: Fine-tune on your data
✓ Compare with Model 1
✓ Explain differences in podcast

---

## Summary: Your Setup

**Model 1 (Yours):** 285M params, trained from scratch
**Model 2 (Better):** 1.5B params (GPT-2 XL), pretrained + fine-tuned

**Training with nohup:** Both scripts support it, won't disconnect!

**Start now:**
```bash
# Model 1 (from scratch)
bash train_nohup.sh

# Model 2 (pretrained) - after Model 1 finishes
nohup python -u llm/scripts/train_model2_pretrained.py > logs/model2.log 2>&1 &
```

Good luck! 🚀
