#!/usr/bin/env python3
"""Debug CNN model outputs"""

import sys
sys.path.insert(0, '/cs/student/projects1/2023/muhamaaz/neural-canvas')

import backend.main as backend
from PIL import Image
import torch
import torch.nn.functional as F

# Load models first
print("Loading models...")
backend.load_models()

# Create test image
test_image = Image.new('RGB', (224, 224), color='blue')
img_tensor = backend.preprocess_image(test_image).cuda()

print("="*60)
print("DEBUGGING CNN MODELS")
print("="*60)

# Check scratch model
print("\n[SCRATCH MODEL]")
print(f"  Model in eval mode: {not backend.cnn_model_scratch.training}")
with torch.no_grad():
    logits = backend.cnn_model_scratch(img_tensor)
    print(f"  Logits shape - artist: {logits['artist'].shape}, style: {logits['style'].shape}, genre: {logits['genre'].shape}")
    print(f"  Artist logits range: [{logits['artist'].min().item():.2f}, {logits['artist'].max().item():.2f}]")
    probs = F.softmax(logits['artist'], dim=1)
    conf, idx = torch.max(probs, dim=1)
    print(f"  Top artist confidence: {conf.item():.1%}, index: {idx.item()}")

# Check fine-tuned model
print("\n[FINE-TUNED MODEL]")
print(f"  Model in eval mode: {not backend.cnn_model_finetuned.training}")
with torch.no_grad():
    logits = backend.cnn_model_finetuned(img_tensor)
    print(f"  Logits shape - artist: {logits['artist'].shape}, style: {logits['style'].shape}, genre: {logits['genre'].shape}")
    print(f"  Artist logits range: [{logits['artist'].min().item():.2f}, {logits['artist'].max().item():.2f}]")
    probs = F.softmax(logits['artist'], dim=1)
    conf, idx = torch.max(probs, dim=1)
    print(f"  Top artist confidence: {conf.item():.1%}, index: {idx.item()}")
    
    # Check if logits are too small (might indicate model not working)
    print(f"  Artist logits std: {logits['artist'].std().item():.2f}")
    print(f"  Top 5 artist probs: {torch.topk(probs, 5).values[0].tolist()}")

print("\n" + "="*60)

