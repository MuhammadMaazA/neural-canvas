#!/usr/bin/env python3
"""Check WikiArt dataset features with streaming"""

from datasets import load_dataset, get_dataset_config_info

print("Getting WikiArt config info...")
try:
    info = get_dataset_config_info("huggan/wikiart", "default")
    print(f"\nFeatures: {info.features}")
except Exception as e:
    print(f"Error getting config: {e}")

print("\nLoading with streaming...")
dataset = load_dataset("huggan/wikiart", split="train", streaming=True)

# Get dataset info
print(f"\nDataset features:")
for name, feature in dataset.features.items():
    print(f"  {name}: {feature}")
    # Check if it's a ClassLabel (categorical) feature
    if hasattr(feature, 'names'):
        print(f"    -> Label names: {feature.names[:10]}...")  # First 10

print("\n" + "="*60)
print("Testing label conversion...")
item = next(iter(dataset))
print(f"\nRaw IDs:")
print(f"  artist: {item['artist']}")
print(f"  genre: {item['genre']}")
print(f"  style: {item['style']}")

# Try to convert IDs to names
if hasattr(dataset.features['artist'], 'int2str'):
    print(f"\nConverted names:")
    print(f"  artist: {dataset.features['artist'].int2str(item['artist'])}")
    print(f"  genre: {dataset.features['genre'].int2str(item['genre'])}")
    print(f"  style: {dataset.features['style'].int2str(item['style'])}")
elif hasattr(dataset.features['artist'], 'names'):
    print(f"\nConverted names:")
    print(f"  artist: {dataset.features['artist'].names[item['artist']]}")
    print(f"  genre: {dataset.features['genre'].names[item['genre']]}")
    print(f"  style: {dataset.features['style'].names[item['style']]}")
