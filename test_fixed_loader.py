#!/usr/bin/env python3
"""Test the fixed WikiArt loader"""

import sys
sys.path.insert(0, '/cs/student/projects1/2023/muhamaaz/neural-canvas')

from llm.utils.curated_art_dataset import load_wikiart_knowledge

print("Testing fixed WikiArt loader...")
print("="*60)

# Load just 100 samples to test
samples = load_wikiart_knowledge(max_samples=100)

print(f"\n✓ Successfully loaded {len(samples)} samples!")
print("\nFirst 5 samples:")
print("="*60)

for i, sample in enumerate(samples[:5]):
    print(f"\n[Sample {i+1}]")
    print(sample)
    print("-"*60)
