"""
Dataset Loader
Loads OpenWebText, C4, WikiArt from HuggingFace
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import List, Dict, Tuple
import re
import random
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer


class TextDataset(Dataset):
    """Tokenized dataset"""
    
    def __init__(self, texts: List[str], tokenizer, max_len: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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


def load_openwebtext(max_samples: int = 500000) -> List[str]:
    """Load text from The Pile"""
    print(f"Loading The Pile ({max_samples:,} samples)...")
    
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    
    texts = []
    for idx, item in enumerate(tqdm(dataset, desc="The Pile", total=max_samples)):
        text = item['text'].strip()
        if len(text) > 200:
            texts.append(text)
        
        if len(texts) >= max_samples:
            break
    
    print(f"Loaded {len(texts):,} texts")
    return texts


def load_c4(max_samples: int = 500000) -> List[str]:
    """Load C4 dataset"""
    print(f"Loading C4 ({max_samples:,} samples)...")
    
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    texts = []
    for idx, item in enumerate(tqdm(dataset, desc="C4", total=max_samples)):
        text = item['text'].strip()
        if len(text) > 150:
            texts.append(text)
        
        if len(texts) >= max_samples:
            break
    
    print(f"Loaded {len(texts):,} texts")
    return texts


def load_wikiart(max_samples: int = 100000) -> List[str]:
    """Load WikiArt dataset"""
    print(f"Loading WikiArt ({max_samples:,} samples)...")
    
    try:
        dataset = load_dataset("huggan/wikiart", split="train")
        
        texts = []
        for item in tqdm(dataset, desc="WikiArt"):
            artist = item.get('artist', 'Unknown')
            style = item.get('style', 'Unknown')
            genre = item.get('genre', 'Unknown')
            
            text = f"Artwork by {artist} in {style} style, {genre} genre."
            texts.append(text)
            
            if len(texts) >= max_samples:
                break
        
        print(f"Loaded {len(texts):,} art descriptions")
        return texts
        
    except Exception as e:
        print(f"Error loading WikiArt: {e}")
        print("Trying alternative art dataset...")
        
        try:
            # Alternative: SBU Captions (image captions)
            dataset = load_dataset("sbu_captions", split="train")
            
            texts = []
            for item in tqdm(dataset, desc="Processing SBU Captions"):
                caption = item['caption'].strip()
                if len(caption) > 20:
                    text = f"This artistic work depicts {caption}, demonstrating creative visual composition and aesthetic principles."
                    texts.append(text)
                
                if len(texts) >= max_samples:
                    break
            
            print(f"Loaded {len(texts):,} art descriptions from SBU Captions")
            return texts
        except:
            print("Using fallback art commentary data...")
            return _fallback_art_data(max_samples)


def build_vocab(texts: List[str], max_vocab: int = 30000, min_freq: int = 5) -> Dict[str, int]:
    """Build vocabulary from texts"""
    print("Building vocabulary...")
    word_counts = Counter()
    
    for text in tqdm(texts, desc="Counting words"):
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts.update(words)
    
    vocab = {
        '<PAD>': 0,
        '<SOS>': 1,
        '<EOS>': 2,
        '<UNK>': 3,
    }
    
    for word, count in word_counts.most_common(max_vocab):
        if count >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
            if len(vocab) >= max_vocab:
                break
    
    print(f"Built vocabulary with {len(vocab):,} tokens")
    return vocab


def load_all_datasets(openwebtext_samples: int = 500000,
                     c4_samples: int = 500000,
                     wikiart_samples: int = 100000) -> List[str]:
    """Load all three datasets"""
    print("=" * 60)
    print("Loading Datasets")
    print("=" * 60)
    
    all_texts = []
    
    dataset_count = sum([1 for x in [openwebtext_samples, c4_samples, wikiart_samples] if x > 0])
    current = 0
    
    if openwebtext_samples > 0:
        current += 1
        print(f"\n[{current}/{dataset_count}] The Pile")
        owt_texts = load_openwebtext(max_samples=openwebtext_samples)
        all_texts.extend(owt_texts)
    else:
        owt_texts = []
    
    if c4_samples > 0:
        current += 1
        print(f"\n[{current}/{dataset_count}] C4")
        c4_texts = load_c4(max_samples=c4_samples)
        all_texts.extend(c4_texts)
    else:
        c4_texts = []
    
    if wikiart_samples > 0:
        current += 1
        print(f"\n[{current}/{dataset_count}] WikiArt")
        art_texts = load_wikiart(max_samples=wikiart_samples)
        all_texts.extend(art_texts)
    else:
        art_texts = []
    
    print("\n" + "=" * 60)
    print(f"Total: {len(all_texts):,} samples")
    if len(owt_texts) > 0:
        print(f"  The Pile: {len(owt_texts):,}")
    if len(c4_texts) > 0:
        print(f"  C4: {len(c4_texts):,}")
    if len(art_texts) > 0:
        print(f"  WikiArt: {len(art_texts):,}")
    print("=" * 60)
    
    return all_texts
