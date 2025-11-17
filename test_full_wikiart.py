#!/usr/bin/env python3
"""Test loading ALL WikiArt samples"""

import sys
sys.path.insert(0, '/cs/student/projects1/2023/muhamaaz/neural-canvas')

from llm.utils.curated_art_dataset import load_wikiart_knowledge

print("Testing WikiArt loader with full dataset...")
print("="*60)

# Load 120K (should get all ~81K available)
samples = load_wikiart_knowledge(max_samples=120000)

print(f"\n✓ Successfully loaded {len(samples):,} samples!")
print(f"\nExpected: ~81,000 samples (full WikiArt dataset)")
print(f"Got: {len(samples):,} samples ({len(samples)/81000*100:.1f}% of expected)")
