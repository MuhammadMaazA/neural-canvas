#!/usr/bin/env python3
"""Check WikiArt dataset features and mappings"""

from datasets import load_dataset

print("Loading WikiArt dataset...")
dataset = load_dataset("huggan/wikiart", split="train", streaming=False)

print(f"\nDataset info:")
print(f"Features: {dataset.features}")

# Check if there are label mappings
if 'artist' in dataset.features:
    print(f"\nArtist feature: {dataset.features['artist']}")
if 'genre' in dataset.features:
    print(f"Genre feature: {dataset.features['genre']}")
if 'style' in dataset.features:
    print(f"Style feature: {dataset.features['style']}")

# Try to get the actual names
print("\nFirst 5 samples:")
for i, item in enumerate(dataset):
    if i >= 5:
        break
    print(f"\nSample {i}:")
    print(f"  Artist ID: {item['artist']}")
    print(f"  Genre ID: {item['genre']}")
    print(f"  Style ID: {item['style']}")
