# CNN Explainer LLM Training

## Overview

This trains LLMs to **explain CNN classification outputs** in natural language. The models learn to:
1. Take CNN outputs (artist/style/genre classifications with confidence scores)
2. Generate accessible explanations of what the CNN detected
3. Explain both the art and the AI's reasoning

## Two Models

| Model | Type | Size | Training |
|-------|------|------|----------|
| **Model 1** | From scratch | 56M params | Custom Art Expert Transformer |
| **Model 2** | Fine-tuned | 124M params | GPT-2 Base pretrained |

## Quick Start

### 1. Test Dataset Generation
```bash
cd /cs/student/projects1/2023/muhamaaz/neural-canvas
source venv/bin/activate
python llm/utils/cnn_explanation_dataset.py
```

### 2. Train From-Scratch Model (Model 1)
```bash
python train_cnn_explainer_from_scratch.py
```

**Expected:**
- Training time: ~4-6 hours on GPU
- 100K training samples (CNN output → explanation pairs)
- 20 epochs, early stopping with patience=5
- Checkpoint dir: `/cs/student/projects1/2023/muhamaaz/checkpoints/cnn_explainer_from_scratch/`

### 3. Fine-tune Pretrained Model (Model 2)
```bash
python finetune_cnn_explainer.py
```

**Expected:**
- Training time: ~2-3 hours on GPU
- Same 100K samples
- 10 epochs, early stopping with patience=3
- Checkpoint dir: `/cs/student/projects1/2023/muhamaaz/checkpoints/cnn_explainer_finetuned/`

## How It Works

### Dataset Format

**Input (CNN Classification Output):**
```
CNN Classification Results:
Artist: Vincent van Gogh (94.3% confidence)
Style: Post-Impressionism (91.7% confidence)
Genre: Landscape (88.5% confidence)

Explain these results:
```

**Target (Natural Explanation):**
```
The neural network identified this artwork with high confidence. The classification
as Post-Impressionism is based on detected visual patterns including vivid colors
and thick application of paint - hallmarks of this movement. The Landscape genre
is evident from natural scenery and outdoor setting that the convolutional layers
identified. Vincent van Gogh is recognized as a master of Post-Impressionism, and
the network's 94.3% confidence suggests the visual features closely match patterns
learned from this artist's known works.
```

### Training Data

Generated synthetically with:
- **Artists**: 25 major artists (Van Gogh, Picasso, Monet, etc.)
- **Styles**: 27 art movements (Impressionism, Cubism, Renaissance, etc.)
- **Genres**: 19 genres (Portrait, Landscape, Still Life, etc.)
- **Confidences**: Realistic distributions (60-98%)
- **Explanations**: 5 different explanation templates

**Result**: 100,000 diverse training pairs

### Model Architecture

**Model 1 (From Scratch):**
- Custom transformer: 8 layers, 512 dim, 8 heads, GQA
- Modern features: RoPE, RMSNorm, SwiGLU
- 56M parameters, optimized for this task

**Model 2 (Fine-tuned):**
- GPT-2 Base (124M params, 12 layers)
- Pretrained on 40GB of text, fine-tuned on CNN explanations
- Lower learning rate (2e-5 vs 3e-4)

## Integration with CNN

### When Your Friend's CNN Runs

Your friend's CNN outputs:
```python
{
    'artist': logits_tensor,   # [batch_size, num_artists]
    'style': logits_tensor,    # [batch_size, num_styles]
    'genre': logits_tensor     # [batch_size, num_genres]
}
```

### Convert to Text for LLM

```python
import torch.nn.functional as F

# Get predictions and confidences
artist_probs = F.softmax(cnn_output['artist'], dim=-1)
style_probs = F.softmax(cnn_output['style'], dim=-1)
genre_probs = F.softmax(cnn_output['genre'], dim=-1)

artist_conf, artist_idx = torch.max(artist_probs, dim=-1)
style_conf, style_idx = torch.max(style_probs, dim=-1)
genre_conf, genre_idx = torch.max(genre_probs, dim=-1)

# Map indices to names
artist_name = ARTIST_CLASSES[artist_idx]
style_name = STYLE_CLASSES[style_idx]
genre_name = GENRE_CLASSES[genre_idx]

# Format for LLM
prompt = f"""CNN Classification Results:
Artist: {artist_name} ({artist_conf:.1%} confidence)
Style: {style_name} ({style_conf:.1%} confidence)
Genre: {genre_name} ({genre_conf:.1%} confidence)

Explain these results:"""

# Generate explanation
explanation = llm.generate(prompt)
```

### Example Usage Script

```python
#!/usr/bin/env python3
"""Test CNN Explainer with Real CNN Outputs"""

import torch
from transformers import GPT2Tokenizer
from llm.models.art_expert_model import ArtExpertTransformer

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('checkpoints/cnn_explainer_from_scratch/best_model.pt')

model = ArtExpertTransformer(**checkpoint['config']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Simulate CNN output
prompt = """CNN Classification Results:
Artist: Claude Monet (96.2% confidence)
Style: Impressionism (94.8% confidence)
Genre: Landscape (91.3% confidence)

Explain these results:"""

# Generate explanation
inputs = tokenizer(prompt, return_tensors='pt').to(device)
with torch.no_grad():
    output_ids = model.generate(
        inputs['input_ids'],
        max_new_tokens=150,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )

explanation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(explanation)
```

## Training Progress

Monitor training with:
```bash
# Watch logs
tail -f /cs/student/projects1/2023/muhamaaz/logs/cnn_explainer_*.log

# View training curves
display checkpoints/cnn_explainer_*/training_curves.png
```

## Checkpoints

Both models save:
- `best_model.pt` - Best validation loss checkpoint
- `latest.pt` - Latest epoch (for resuming)
- `training_curves.png` - Loss/perplexity plots

Model 2 also saves:
- `best_model_hf/` - HuggingFace format (easy loading)

## Comparison: Model 1 vs Model 2

| Aspect | Model 1 (Scratch) | Model 2 (Fine-tuned) |
|--------|------------------|---------------------|
| **Speed** | Slower (~6 hrs) | Faster (~3 hrs) |
| **Quality (initial)** | Lower | Higher (pretrained) |
| **Task-specific** | Fully optimized | General → Specialized |
| **Size** | 56M | 82M |
| **Best for** | Understanding training | Quick deployment |

**For your project**: Train BOTH to compare!

## Expected Results

After training, models should generate explanations like:

**Good Output:**
> "The AI identified this as Impressionism with 94.8% confidence by detecting visible
> brushstrokes and emphasis on light - signature characteristics of this movement.
> The Landscape genre is evident from the outdoor scenery composition. Claude Monet,
> a founding Impressionist, is attributed with 96.2% confidence based on distinctive
> visual patterns the network learned from his known works."

**Avoid (current old model):**
> "worked within the Impressionism tradition, creating Landscape pieces that showcased..."

## Troubleshooting

### OOM (Out of Memory)
Reduce batch size in training scripts:
```python
BATCH_SIZE = 8  # Try 4 if OOM
GRADIENT_ACCUMULATION_STEPS = 4  # Increase to 8
```

### Slow Training
Enable mixed precision (should be on by default):
```python
MIXED_PRECISION = True
```

### Poor Quality
- Check dataset generation works correctly
- Ensure 100K samples are loaded
- Verify loss is decreasing
- Wait for at least 10 epochs

## Files Created

```
neural-canvas/
├── llm/
│   └── utils/
│       └── cnn_explanation_dataset.py    # Dataset generator
├── train_cnn_explainer_from_scratch.py   # Model 1 training
├── finetune_cnn_explainer.py             # Model 2 training
└── README_CNN_EXPLAINER.md               # This file
```

## Next Steps

1. ✅ Test dataset generation
2. ✅ Start training (choose one or both)
3. 🔄 Monitor training logs
4. ✅ Load best checkpoint
5. ✅ Integrate with CNN outputs
6. ✅ Test with real classifications

## Questions?

The models are configured for your hardware and dataset. Both should work out of the box:

```bash
# From scratch (56M params, slower but tailored)
python train_cnn_explainer_from_scratch.py

# Fine-tuned (82M params, faster with pretrained knowledge)
python finetune_cnn_explainer.py
```

Training progress saves automatically. You can stop/resume anytime.

---

**Goal**: Train LLMs that can look at CNN classification results and explain them in natural, educational language - making both art and AI accessible to everyone! 🎨🤖
