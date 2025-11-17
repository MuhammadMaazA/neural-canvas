#!/usr/bin/env python3
"""Test the enhanced art dataset with rich knowledge"""

import sys
sys.path.insert(0, '/cs/student/projects1/2023/muhamaaz/neural-canvas')

from llm.utils.curated_art_dataset import load_curated_art_datasets

print("Testing enhanced dataset loading...")
print("=" * 60)

# Load small sample to test
texts, stats = load_curated_art_datasets(
    art_knowledge=500,  # Test with 500 samples
    ai_literacy=100,
    conversational=100
)

print("\n✅ Dataset loaded successfully!")
print(f"Total: {len(texts)} samples")
print(f"Stats: {stats}")

print("\n" + "=" * 60)
print("Sample Art Knowledge Examples:")
print("=" * 60)

# Show some rich knowledge samples
for i, text in enumerate(texts[:5]):
    if 'Van Gogh' in text or 'Impressionism' in text or 'chiaroscuro' in text:
        print(f"\n[Sample {i+1}]")
        print(text[:500] + "..." if len(text) > 500 else text)
        print("-" * 60)
