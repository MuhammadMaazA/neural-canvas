"""
Dataset utilities for loading Cornell, ELI5, and Technical Q&A datasets
UCL COMP0220 Coursework - Complete Implementation
"""

import torch
from torch.utils.data import Dataset
from collections import Counter
import requests
import zipfile
import os
import json
import re
from tqdm import tqdm
from typing import List, Dict, Tuple

class TextDataset(Dataset):
    """PyTorch Dataset for text sequences"""
    
    def __init__(self, texts: List[str], vocab: Dict[str, int], max_len: int = 256):
        """
        Args:
            texts: List of text strings
            vocab: Vocabulary dictionary mapping words to indices
            max_len: Maximum sequence length
        """
        self.texts = texts
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns tokenized input and target (same for language modeling)"""
        text = self.texts[idx]
        tokens = self._tokenize(text)
        
        # Pad or truncate to max_len
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        
        tokens = torch.tensor(tokens[:self.max_len], dtype=torch.long)
        
        # For language modeling, input and target are the same (shifted by 1 in model)
        return tokens, tokens
    
    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text into vocabulary indices"""
        # Simple word-level tokenization with basic cleaning
        text = text.lower().strip()
        words = re.findall(r'\b\w+\b', text)
        tokens = [self.vocab.get(word, 3) for word in words]  # 3 = <UNK>
        return tokens


def download_cornell() -> str:
    """Download Cornell Movie Dialogs Corpus"""
    url = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
    corpus_dir = "cornell_movie_dialogs_corpus"
    
    if os.path.exists(corpus_dir):
        return corpus_dir
    
    print("Downloading Cornell Movie Dialogs Corpus...")
    zip_path = "cornell.zip"
    
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        
        total_size = int(r.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall()
        
        os.remove(zip_path)
        print("Cornell dataset ready!")
        
    except Exception as e:
        print(f"Error downloading Cornell dataset: {e}")
        raise
    
    return corpus_dir


def load_cornell(max_samples: int = 100000) -> List[str]:
    """
    Load Cornell Movie Dialogs into conversation pairs
    Args:
        max_samples: Maximum number of conversation pairs to load
    Returns:
        List of conversation strings (question + answer pairs)
    """
    print(f"Loading Cornell Movie Dialogs (max {max_samples:,} samples)...")
    base = download_cornell()
    
    lines = {}
    lines_file = None
    conv_file = None
    
    # Debug: List all files to see what we have
    print("Searching for Cornell dataset files...")
    all_files = []
    for root, dirs, files in os.walk(base):
        for f in files:
            filepath = os.path.join(root, f)
            all_files.append(filepath)
            if 'movie_lines' in f.lower() or 'lines' in f.lower():
                if lines_file is None:
                    lines_file = filepath
                    print(f"Found lines file: {lines_file}")
            if 'movie_conversations' in f.lower() or 'conversations' in f.lower():
                if conv_file is None:
                    conv_file = filepath
                    print(f"Found conversations file: {conv_file}")
    
    # If still not found, list all files for debugging
    if not lines_file or not conv_file:
        print(f"\nAvailable files in {base}:")
        for f in all_files[:20]:  # Show first 20 files
            print(f"  {f}")
        if len(all_files) > 20:
            print(f"  ... and {len(all_files) - 20} more files")
    
    if not lines_file:
        # Try alternative names
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith('.txt') and ('line' in f.lower() or 'utterance' in f.lower()):
                    lines_file = os.path.join(root, f)
                    print(f"Using alternative lines file: {lines_file}")
                    break
            if lines_file:
                break
    
    if not lines_file:
        raise FileNotFoundError(f"movie_lines file not found in Cornell corpus. Searched in: {base}")
    
    print("Reading movie lines...")
    with open(lines_file, 'r', encoding='iso-8859-1', errors='ignore') as file:
        for line in tqdm(file, desc="Processing lines"):
            parts = line.split(' +++$+++ ')
            if len(parts) >= 5:
                line_id = parts[0]
                text = parts[4].strip()
                lines[line_id] = text
    
    print(f"Loaded {len(lines):,} movie lines")
    
    # Find and read movie_conversations file (if not already found)
    conversations = []
    if not conv_file:
        for root, _, files in os.walk(base):
            for f in files:
                if 'movie_conversations' in f.lower() or 'conversations' in f.lower():
                    conv_file = os.path.join(root, f)
                    print(f"Found conversations file: {conv_file}")
                    break
            if conv_file:
                break
    
    if not conv_file:
        # Try alternative names
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith('.txt') and ('conv' in f.lower() or 'dialogue' in f.lower()):
                    conv_file = os.path.join(root, f)
                    print(f"Using alternative conversations file: {conv_file}")
                    break
            if conv_file:
                break
    
    if not conv_file:
        raise FileNotFoundError(f"movie_conversations file not found in Cornell corpus. Searched in: {base}")
    
    print("Building conversation pairs...")
    with open(conv_file, 'r', encoding='iso-8859-1', errors='ignore') as file:
        for line in tqdm(file, desc="Processing conversations"):
            parts = line.split(' +++$+++ ')
            if len(parts) >= 4:
                try:
                    # Extract line IDs from the conversation
                    ids_str = parts[3].strip()
                    ids = eval(ids_str)  # Safe eval for list of strings
                    
                    # Create pairs from consecutive lines
                    for i in range(len(ids) - 1):
                        if ids[i] in lines and ids[i+1] in lines:
                            # Combine question and answer
                            pair = f"{lines[ids[i]]} {lines[ids[i+1]]}"
                            conversations.append(pair)
                            
                            if len(conversations) >= max_samples:
                                break
                    
                    if len(conversations) >= max_samples:
                        break
                except:
                    continue
    
    print(f"Loaded {len(conversations):,} conversation pairs from Cornell")
    return conversations[:max_samples]


def load_eli5(max_samples: int = 50000) -> List[str]:
    """
    Load ELI5 (Explain Like I'm 5) dataset from Hugging Face
    Falls back to synthetic data if Hugging Face unavailable
    """
    print(f"Loading ELI5 dataset (max {max_samples:,} samples)...")
    
    try:
        from datasets import load_dataset
        
        print("Attempting to load ELI5 from Hugging Face...")
        dataset = load_dataset("eli5", split="train_eli5", streaming=False)
        
        texts = []
        for item in tqdm(dataset, desc="Processing ELI5", total=min(max_samples, len(dataset))):
            # Combine question and answer
            question = item.get('title', '') + ' ' + item.get('selftext', '')
            answer = ' '.join([a.get('text', '') for a in item.get('answers', {}).get('text', [])[:1]])
            
            if question and answer:
                combined = f"{question} {answer}"
                texts.append(combined)
                
                if len(texts) >= max_samples:
                    break
        
        print(f"Loaded {len(texts):,} samples from ELI5")
        return texts
        
    except Exception as e:
        print(f"Could not load ELI5 from Hugging Face: {e}")
        print("Using curated AI literacy dataset instead...")
        
        # Curated AI literacy Q&A pairs
        ai_literacy_pairs = [
            ("what is artificial intelligence", "artificial intelligence is when computers learn and solve problems like humans"),
            ("how do computers learn", "computers learn by analyzing data finding patterns and adjusting their algorithms"),
            ("what is a neural network", "a neural network is a system inspired by the brain with layers of connected nodes"),
            ("what is machine learning", "machine learning is when computers improve automatically through experience"),
            ("how does deep learning work", "deep learning uses many layers of neural networks to learn complex patterns"),
            ("what is backpropagation", "backpropagation is how neural networks learn by calculating errors and updating weights"),
            ("what is overfitting", "overfitting is when a model memorizes training data but fails on new data"),
            ("what is a gradient", "a gradient shows how to adjust parameters to reduce error"),
            ("what is a loss function", "a loss function measures how wrong a model's predictions are"),
            ("what is regularization", "regularization prevents overfitting by adding constraints to the model"),
            ("what is transfer learning", "transfer learning uses knowledge from one task to help with another"),
            ("what is a transformer", "a transformer is a neural network architecture that processes sequences using attention"),
            ("what is attention in ai", "attention lets models focus on important parts of input when making predictions"),
            ("what is a chatbot", "a chatbot is a computer program that talks with humans using natural language"),
            ("what is natural language processing", "natural language processing helps computers understand and generate human language"),
            ("what is computer vision", "computer vision helps computers understand and analyze images"),
            ("what is reinforcement learning", "reinforcement learning trains agents through rewards and punishments"),
            ("what is supervised learning", "supervised learning uses labeled examples to train models"),
            ("what is unsupervised learning", "unsupervised learning finds patterns in data without labels"),
            ("what is data augmentation", "data augmentation creates new training examples by modifying existing ones"),
        ]
        
        # Expand with variations
        texts = []
        for question, answer in ai_literacy_pairs:
            texts.append(f"{question} {answer}")
            # Add variations
            texts.append(f"explain {question} {answer}")
            texts.append(f"tell me about {question} {answer}")
            texts.append(f"how does {question} work {answer}")
        
        # Repeat to reach target size
        multiplier = max_samples // len(texts) + 1
        texts = (texts * multiplier)[:max_samples]
        
        print(f"Created {len(texts):,} AI literacy samples")
        return texts


def load_technical_qa(max_samples: int = 30000) -> List[str]:
    """
    Load Technical Deep Learning Q&A dataset
    Creates curated technical questions and answers
    """
    print(f"Loading Technical Q&A dataset (max {max_samples:,} samples)...")
    
    technical_pairs = [
        ("backpropagation algorithm", "backpropagation calculates gradients using chain rule to update neural network weights"),
        ("dropout regularization", "dropout randomly disables neurons during training to prevent overfitting"),
        ("batch normalization", "batch normalization normalizes layer inputs to stabilize and accelerate training"),
        ("learning rate scheduling", "learning rate scheduling adjusts learning rate during training for better convergence"),
        ("gradient clipping", "gradient clipping limits gradient values to prevent exploding gradients"),
        ("adam optimizer", "adam optimizer combines momentum and adaptive learning rates for efficient optimization"),
        ("cross entropy loss", "cross entropy loss measures difference between predicted and true probability distributions"),
        ("attention mechanism", "attention mechanism allows models to focus on relevant parts of input sequences"),
        ("transformer architecture", "transformer architecture uses self attention and feed forward networks for sequence processing"),
        ("layer normalization", "layer normalization normalizes activations across features for each sample"),
        ("residual connections", "residual connections add skip connections to help gradients flow through deep networks"),
        ("convolutional neural network", "convolutional neural networks use filters to detect patterns in images"),
        ("recurrent neural network", "recurrent neural networks process sequences by maintaining hidden state"),
        ("long short term memory", "lstm networks use gated cells to remember information over long sequences"),
        ("gated recurrent unit", "gru networks use simpler gating than lstm for efficient sequence modeling"),
        ("word embeddings", "word embeddings represent words as dense vectors capturing semantic meaning"),
        ("tokenization", "tokenization splits text into smaller units like words or subwords for processing"),
        ("fine tuning", "fine tuning adapts pre trained models to specific tasks using transfer learning"),
        ("data preprocessing", "data preprocessing cleans and transforms raw data for model training"),
        ("hyperparameter tuning", "hyperparameter tuning finds optimal model settings through systematic search"),
    ]
    
    texts = []
    for question, answer in technical_pairs:
        texts.append(f"{question} {answer}")
        texts.append(f"what is {question} {answer}")
        texts.append(f"explain {question} {answer}")
        texts.append(f"how does {question} work {answer}")
    
    multiplier = max_samples // len(texts) + 1
    texts = (texts * multiplier)[:max_samples]
    
    print(f"Created {len(texts):,} technical Q&A samples")
    return texts


def build_vocab(texts: List[str], max_vocab: int = 30000, min_freq: int = 5) -> Dict[str, int]:
    """
    Build vocabulary from texts
    Args:
        texts: List of text strings
        max_vocab: Maximum vocabulary size
        min_freq: Minimum word frequency to include
    Returns:
        Vocabulary dictionary mapping words to indices
    """
    print("Building vocabulary...")
    word_counts = Counter()
    
    for text in tqdm(texts, desc="Counting words"):
        # Simple word tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts.update(words)
    
    # Build vocab with special tokens
    vocab = {
        '<PAD>': 0,  # Padding token
        '<SOS>': 1,  # Start of sequence
        '<EOS>': 2,  # End of sequence
        '<UNK>': 3,  # Unknown word
    }
    
    # Add most frequent words that meet minimum frequency
    for word, count in word_counts.most_common(max_vocab):
        if count >= min_freq and word not in vocab:
            vocab[word] = len(vocab)
            if len(vocab) >= max_vocab:
                break
    
    print(f"Built vocabulary with {len(vocab):,} tokens")
    print(f"  Most common words: {list(word_counts.most_common(10))}")
    
    return vocab


def load_all_datasets(cornell_samples: int = 100000, eli5_samples: int = 50000, tech_samples: int = 30000) -> List[str]:
    """
    Load all three datasets and combine them
    Args:
        cornell_samples: Number of Cornell samples to load
        eli5_samples: Number of ELI5 samples to load
        tech_samples: Number of Technical Q&A samples to load
    Returns:
        Combined list of all text samples
    """
    print("=" * 80)
    print("LOADING ALL DATASETS")
    print("=" * 80)
    
    all_texts = []
    
    # Dataset 1: Cornell Movie Dialogs
    print("\n[1/3] Cornell Movie Dialogs Corpus")
    cornell_texts = load_cornell(max_samples=cornell_samples)
    all_texts.extend(cornell_texts)
    
    # Dataset 2: ELI5
    print("\n[2/3] ELI5 (Explain Like I'm 5) Dataset")
    eli5_texts = load_eli5(max_samples=eli5_samples)
    all_texts.extend(eli5_texts)
    
    # Dataset 3: Technical Q&A
    print("\n[3/3] Technical Deep Learning Q&A")
    tech_texts = load_technical_qa(max_samples=tech_samples)
    all_texts.extend(tech_texts)
    
    print("\n" + "=" * 80)
    print(f"TOTAL: {len(all_texts):,} samples")
    print(f"  Cornell: {len(cornell_texts):,}")
    print(f"  ELI5: {len(eli5_texts):,}")
    print(f"  Technical: {len(tech_texts):,}")
    print("=" * 80)
    
    return all_texts

