#!/usr/bin/env python3
"""Test script to verify CNN model outputs"""

import sys
sys.path.insert(0, '/cs/student/projects1/2023/muhamaaz/neural-canvas')

from backend.main import predict_cnn, load_models, preprocess_image
from PIL import Image
import torch

# Load models
print("Loading models...")
load_models()

# Create a test image (simple colored square)
test_image = Image.new('RGB', (224, 224), color='red')

print("\n" + "="*60)
print("TESTING CNN MODELS")
print("="*60)

# Test scratch model
print("\n[SCRATCH MODEL - The Novice]")
try:
    scratch_pred = predict_cnn(test_image, "scratch")
    print(f"  Artist: {scratch_pred.artist} ({scratch_pred.artist_confidence:.1%})")
    print(f"  Style: {scratch_pred.style} ({scratch_pred.style_confidence:.1%})")
    print(f"  Genre: {scratch_pred.genre} ({scratch_pred.genre_confidence:.1%})")
except Exception as e:
    print(f"  ERROR: {e}")

# Test fine-tuned model
print("\n[FINE-TUNED MODEL - The Expert]")
try:
    finetuned_pred = predict_cnn(test_image, "finetuned")
    print(f"  Artist: {finetuned_pred.artist} ({finetuned_pred.artist_confidence:.1%})")
    print(f"  Style: {finetuned_pred.style} ({finetuned_pred.style_confidence:.1%})")
    print(f"  Genre: {finetuned_pred.genre} ({finetuned_pred.genre_confidence:.1%})")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "="*60)
print("VERIFICATION:")
print("  - Scratch model should show lower confidence (34% val acc)")
print("  - Fine-tuned model should show higher confidence (87% val acc)")
print("="*60)

