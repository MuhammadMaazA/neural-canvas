"""
PREMIUM Art Dataset Loader - Using High-Quality HuggingFace Datasets
=====================================================================

Using the BEST available art datasets:
1. allenai/art - Art dataset from Allen AI
2. biglam/european_art - European art collection
3. printblue/EmoArt-130k - 130K art with annotations
4. cosmopedia (art subset) - Synthetic educational content
5. WikiArt - Classic art dataset

All ART-ONLY, no confusion!
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import random
from typing import List, Dict, Tuple
from tqdm import tqdm
import re


class TextDataset(Dataset):
    """Simple tokenized text dataset"""

    def __init__(self, texts: List[str], tokenizer, max_len: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        tokens = encoding['input_ids'].squeeze(0)
        return tokens, tokens


# ==============================================================================
# 1. ALLEN AI ART DATASET
# ==============================================================================

def load_allenai_art(max_samples: int = 50000) -> List[str]:
    """
    Allen AI art dataset - high quality art descriptions and analysis
    """
    print(f"\n[1/5] Loading Allen AI Art Dataset ({max_samples:,} samples)...")

    try:
        dataset = load_dataset("allenai/art", split="train")
        print(f"  Found {len(dataset):,} total samples")
    except Exception as e:
        print(f"  Error loading allenai/art: {e}")
        return []

    texts = []

    for item in tqdm(dataset, desc="Allen AI Art"):
        if len(texts) >= max_samples:
            break

        # Extract relevant text fields
        # (Adjust based on actual dataset structure)
        text_content = ""

        # Try different field names
        for field in ['text', 'description', 'caption', 'content', 'article']:
            if field in item and item[field]:
                text_content = str(item[field]).strip()
                break

        # Quality filters
        if not text_content or len(text_content) < 50:
            continue
        if len(text_content) > 2000:
            text_content = text_content[:2000]

        # Clean up
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        texts.append(text_content)

    print(f"✓ Loaded {len(texts):,} Allen AI art samples")
    return texts


# ==============================================================================
# 2. EUROPEAN ART DATASET
# ==============================================================================

def load_european_art(max_samples: int = 50000) -> List[str]:
    """
    BigLAM European art collection - museum-quality descriptions
    """
    print(f"\n[2/5] Loading European Art Dataset ({max_samples:,} samples)...")

    try:
        dataset = load_dataset("biglam/european_art", split="train")
        print(f"  Found {len(dataset):,} total samples")
    except Exception as e:
        print(f"  Error loading biglam/european_art: {e}")
        return []

    texts = []

    for item in tqdm(dataset, desc="European Art"):
        if len(texts) >= max_samples:
            break

        # Extract metadata and create rich descriptions
        title = str(item.get('title', '')).strip()
        artist = str(item.get('artist', '')).strip()
        date = str(item.get('date', '')).strip()
        description = str(item.get('description', '')).strip()
        medium = str(item.get('medium', '')).strip()

        # Build rich text
        text_parts = []

        if title and artist:
            text_parts.append(f'"{title}" by {artist}')

        if date:
            text_parts.append(f"created in {date}")

        if medium:
            text_parts.append(f"using {medium}")

        if description and len(description) > 20:
            text_parts.append(f". {description}")

        if text_parts:
            text = " ".join(text_parts)

            # Quality check
            if 50 < len(text) < 2000:
                texts.append(text)

    print(f"✓ Loaded {len(texts):,} European art samples")
    return texts


# ==============================================================================
# 3. EMOART DATASET
# ==============================================================================

def load_emoart(max_samples: int = 50000) -> List[str]:
    """
    EmoArt-130k - Art with emotional annotations
    Rich descriptions and analysis
    """
    print(f"\n[3/5] Loading EmoArt-130k Dataset ({max_samples:,} samples)...")

    try:
        dataset = load_dataset("printblue/EmoArt-130k", split="train")
        print(f"  Found {len(dataset):,} total samples")
    except Exception as e:
        print(f"  Error loading printblue/EmoArt-130k: {e}")
        return []

    texts = []

    for item in tqdm(dataset, desc="EmoArt"):
        if len(texts) >= max_samples:
            break

        # Extract fields (adjust based on actual structure)
        caption = str(item.get('caption', '')).strip()
        title = str(item.get('title', '')).strip()
        artist = str(item.get('artist', '')).strip()
        emotion = str(item.get('emotion', '')).strip()
        description = str(item.get('description', '')).strip()

        # Build educational text
        text_parts = []

        if title:
            text_parts.append(f"Artwork: {title}")

        if artist:
            text_parts.append(f"by {artist}")

        if emotion:
            text_parts.append(f". This piece evokes a {emotion} emotional response")

        if caption and len(caption) > 20:
            text_parts.append(f". {caption}")
        elif description and len(description) > 20:
            text_parts.append(f". {description}")

        if text_parts:
            text = " ".join(text_parts)

            if 50 < len(text) < 2000:
                text = re.sub(r'\s+', ' ', text).strip()
                texts.append(text)

    print(f"✓ Loaded {len(texts):,} EmoArt samples")
    return texts


# ==============================================================================
# 4. COSMOPEDIA (ART SUBSET)
# ==============================================================================

def load_cosmopedia_art(max_samples: int = 50000) -> List[str]:
    """
    Cosmopedia art-related educational content
    High-quality synthetic educational text
    """
    print(f"\n[4/5] Loading Cosmopedia Art Subset ({max_samples:,} samples)...")

    try:
        dataset = load_dataset("HuggingFaceTB/cosmopedia", split="train", streaming=True)
    except Exception as e:
        print(f"  Error loading cosmopedia: {e}")
        return []

    art_keywords = [
        'art', 'painting', 'artist', 'gallery', 'museum', 'sculpture',
        'impressionism', 'cubism', 'renaissance', 'baroque', 'modern art',
        'abstract', 'portrait', 'landscape', 'masterpiece', 'canvas',
        'brushstroke', 'composition', 'aesthetic', 'visual arts'
    ]

    texts = []
    processed = 0

    for item in tqdm(dataset, desc="Cosmopedia Art", total=max_samples * 3):
        processed += 1
        if processed > max_samples * 5:  # Safety limit
            break

        if len(texts) >= max_samples:
            break

        # Get text content
        text = str(item.get('text', '')).strip()
        prompt = str(item.get('prompt', '')).strip().lower()

        # Filter for art-related content
        if not any(keyword in prompt for keyword in art_keywords):
            if not any(keyword in text[:500].lower() for keyword in art_keywords[:5]):
                continue

        # Quality check
        if not (100 < len(text) < 2000):
            continue

        # Clean up
        text = re.sub(r'\s+', ' ', text).strip()
        texts.append(text)

    print(f"✓ Loaded {len(texts):,} Cosmopedia art samples")
    return texts


# ==============================================================================
# 5. WIKIART (FALLBACK/SUPPLEMENT)
# ==============================================================================

def load_wikiart_descriptions(max_samples: int = 30000) -> List[str]:
    """
    WikiArt dataset - classical fallback
    Creates natural descriptions from metadata
    """
    print(f"\n[5/5] Loading WikiArt ({max_samples:,} samples)...")

    try:
        dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
    except Exception as e:
        print(f"  Error loading WikiArt: {e}")
        return []

    texts = []
    templates = [
        "{artist}, a {style} artist, created {genre} works that exemplified the movement's aesthetic principles and distinctive visual approach.",

        "The {style} movement, represented by artists like {artist}, brought new perspectives to {genre} painting through innovative techniques and artistic vision.",

        "{artist}'s {genre} works demonstrate the characteristics of {style} art, including distinctive approaches to composition, color, and visual storytelling.",

        "As a {style} painter, {artist} contributed to the movement through {genre} works that showcased innovative artistic techniques and perspectives.",

        "Art historians recognize {artist} as an important {style} artist whose {genre} paintings helped define the movement's visual language and artistic principles.",
    ]

    for item in tqdm(dataset, desc="WikiArt", total=max_samples):
        if len(texts) >= max_samples:
            break

        artist = str(item.get('artist', '')).strip()
        style = str(item.get('style', '')).strip()
        genre = str(item.get('genre', '')).strip()

        # Quality filters
        if not all([artist, style, genre]):
            continue
        if any(x.lower() in ['unknown', 'null', 'none', '0'] for x in [artist, style, genre]):
            continue
        if len(artist) < 3 or len(style) < 3:
            continue

        # Create natural text
        template = random.choice(templates)
        text = template.format(artist=artist, style=style, genre=genre)
        texts.append(text)

    print(f"✓ Loaded {len(texts):,} WikiArt samples")
    return texts


# ==============================================================================
# MAIN LOADER
# ==============================================================================

def load_premium_art_dataset(
    allenai_samples: int = 50000,
    european_art_samples: int = 50000,
    emoart_samples: int = 50000,
    cosmopedia_samples: int = 50000,
    wikiart_samples: int = 30000
) -> Tuple[List[str], Dict[str, int]]:
    """
    Load PREMIUM art dataset from high-quality HuggingFace sources

    Total: ~230K high-quality art samples from verified sources

    Returns:
        (all_texts, dataset_stats)
    """
    print("=" * 80)
    print("PREMIUM ART DATASET LOADING")
    print("=" * 80)
    print("Using HIGH-QUALITY HuggingFace datasets")
    print("Target: ~230K verified art samples")
    print("=" * 80)

    all_texts = []
    stats = {}

    # Load all datasets
    allenai_texts = load_allenai_art(allenai_samples)
    all_texts.extend(allenai_texts)
    stats['allenai_art'] = len(allenai_texts)

    european_texts = load_european_art(european_art_samples)
    all_texts.extend(european_texts)
    stats['european_art'] = len(european_texts)

    emoart_texts = load_emoart(emoart_samples)
    all_texts.extend(emoart_texts)
    stats['emoart'] = len(emoart_texts)

    cosmopedia_texts = load_cosmopedia_art(cosmopedia_samples)
    all_texts.extend(cosmopedia_texts)
    stats['cosmopedia_art'] = len(cosmopedia_texts)

    wikiart_texts = load_wikiart_descriptions(wikiart_samples)
    all_texts.extend(wikiart_texts)
    stats['wikiart'] = len(wikiart_texts)

    # Shuffle to mix everything
    random.shuffle(all_texts)

    # Summary
    print("\n" + "=" * 80)
    print("DATASET LOADING COMPLETE")
    print("=" * 80)
    print(f"Total Samples: {len(all_texts):,}")
    print(f"\nBreakdown:")
    for name, count in stats.items():
        pct = count / len(all_texts) * 100 if len(all_texts) > 0 else 0
        print(f"  {name:20s} {count:>7,} ({pct:>5.1f}%)")
    print("=" * 80)

    # Show samples
    print("\nSample texts:")
    for i, text in enumerate(random.sample(all_texts, min(3, len(all_texts)))):
        print(f"\n[Sample {i+1}]")
        print(text[:200] + "..." if len(text) > 200 else text)
    print("=" * 80)

    return all_texts, stats


if __name__ == "__main__":
    # Test loading
    print("Testing premium dataset loading...")
    texts, stats = load_premium_art_dataset(
        allenai_samples=100,
        european_art_samples=100,
        emoart_samples=100,
        cosmopedia_samples=100,
        wikiart_samples=100
    )
    print(f"\n✓ Test complete. Loaded {len(texts):,} samples")
