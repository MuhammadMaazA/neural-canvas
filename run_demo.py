#!/usr/bin/env python3
"""
Quick Demo - Run this to test both LLM models
"""
import os
os.environ['HF_HOME'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append('/cs/student/projects1/2023/muhamaaz/neural-canvas/llm')
from models.art_expert_model import create_art_expert_model

print("\n🔄 Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load models
checkpoint = torch.load("/cs/student/projects1/2023/muhamaaz/checkpoints/cnn_explainer_from_scratch/best_model.pt", map_location=device, weights_only=False)
model1 = create_art_expert_model(tokenizer.vocab_size, "base").to(device)
model1.load_state_dict(checkpoint['model_state_dict'])
model1.eval()

model2 = AutoModelForCausalLM.from_pretrained("/cs/student/projects1/2023/muhamaaz/checkpoints/cnn_explainer_finetuned/best_model_hf").to(device)
model2.eval()

print("✅ Models loaded!\n")

# Test samples
samples = [
    ("Vincent van Gogh", "Post-Impressionism", "Landscape", 0.87, 0.92, 0.78),
    ("Pablo Picasso", "Cubism", "Portrait", 0.76, 0.88, 0.72),
    ("Claude Monet", "Impressionism", "Landscape", 0.91, 0.95, 0.85),
]

for artist, style, genre, a, s, g in samples:
    prompt = f"""The CNN classified this artwork as:
- Artist: {artist} ({a:.0%} confidence)
- Style: {style} ({s:.0%} confidence)
- Genre: {genre} ({g:.0%} confidence)

Explain this classification:"""

    print("="*70)
    print(f"📊 CNN: {artist} | {style} | {genre}")
    print("="*70)
    
    tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    
    with torch.no_grad():
        out1 = model1.generate(tokens, max_new_tokens=100, temperature=0.7)
        out2 = model2.generate(tokens, max_new_tokens=100, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    
    resp1 = tokenizer.decode(out1[0], skip_special_tokens=True)[len(prompt):].strip()
    resp2 = tokenizer.decode(out2[0], skip_special_tokens=True)[len(prompt):].strip()
    
    print("\n🤖 MODEL 1 (From Scratch):")
    print(resp1[:300])
    print("\n🎯 MODEL 2 (Fine-tuned):")
    print(resp2[:300])
    print()

print("✅ Demo complete!")



