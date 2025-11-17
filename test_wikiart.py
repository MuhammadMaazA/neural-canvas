#!/usr/bin/env python3
"""Test WikiArt dataset loading"""

from datasets import load_dataset

print("Loading WikiArt dataset...")
try:
    dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
    print("✓ Dataset loaded successfully!")
    
    # Get first item
    item = next(iter(dataset))
    print(f"\nFields available: {list(item.keys())}")
    print(f"\nSample item:")
    for key, value in item.items():
        if key == 'image':
            print(f"  {key}: <PIL.Image>")
        else:
            print(f"  {key}: {value}")
    
except Exception as e:
    print(f"✗ Error loading WikiArt: {e}")
    import traceback
    traceback.print_exc()
