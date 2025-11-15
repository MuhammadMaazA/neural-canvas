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
    """Tokenized dataset with data augmentation to reduce overfitting"""
    
    def __init__(self, texts: List[str], tokenizer, max_len: int = 512, augment: bool = True):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def _augment_text(self, text: str) -> str:
        """Apply random text augmentation to reduce overfitting"""
        if not self.augment or random.random() > 0.2:  # Only 20% chance (was 30%)
            return text
        
        # Random dropout: randomly drop only 3-5% of words (was 5-10%)
        words = text.split()
        if len(words) > 10 and random.random() < 0.3:  # Less frequent (was 0.5)
            drop_ratio = random.uniform(0.03, 0.05)  # Less aggressive
            num_keep = max(5, int(len(words) * (1 - drop_ratio)))
            indices = sorted(random.sample(range(len(words)), num_keep))
            words = [words[i] for i in indices]
            text = ' '.join(words)
        
        return text
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        
        # Apply augmentation during training
        text = self._augment_text(text)
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        tokens = encoding['input_ids'].squeeze(0)
        return tokens, tokens


def load_squad_qa(max_samples: int = 100000) -> List[str]:
    """Load SQuAD dataset for AI literacy Q&A - coursework aligned"""
    print(f"Loading SQuAD Q&A Dataset ({max_samples:,} samples)...")
    
    # Use SQuAD - Stanford Question Answering Dataset
    dataset = load_dataset("squad", split="train")
    
    texts = []
    for item in tqdm(dataset, desc="SQuAD Q&A"):
        question = item.get('question', '').strip()
        context = item.get('context', '').strip()
        answers = item.get('answers', {}).get('text', [])
        
        if question and answers and len(answers) > 0:
            answer = answers[0].strip()
            
            # Create Q&A format for AI literacy training
            if len(answer) > 20 and len(question) > 10:
                text = f"Question: {question}\n\nAnswer: {answer}\n\nContext: {context[:400]}"
                texts.append(text)
        
        if len(texts) >= max_samples:
            break
    
    print(f"Loaded {len(texts):,} Q&A pairs for AI literacy")
    return texts


def load_ai_ml_qa(max_samples: int = 30000) -> List[str]:
    """Load technical programming Q&A - coursework aligned"""
    print(f"Loading Technical Q&A Dataset ({max_samples:,} samples)...")
    
    # StackOverflow Q&A - programming/tech questions (Parquet format)
    dataset = load_dataset("koutch/stackoverflow_python", split="train")
    
    texts = []
    for item in tqdm(dataset, desc="Technical Q&A"):
        # Try different possible field names
        question = item.get('question', item.get('question_body', item.get('title', ''))).strip()
        answer = item.get('answer', item.get('answer_body', item.get('body', ''))).strip()
        
        # More lenient filtering
        if question and answer and len(question) > 10 and len(answer) > 30:
            # Truncate if too long
            if len(answer) > 1500:
                answer = answer[:1500] + "..."
            formatted = f"Question: {question}\n\nAnswer: {answer}"
            texts.append(formatted)
        
        if len(texts) >= max_samples:
            break
    
    print(f"Loaded {len(texts):,} technical Q&A")
    return texts


def load_conversational_ai(max_samples: int = 50000) -> List[str]:
    """Load conversational dialogues for podcast-style AI literacy training - coursework aligned"""
    print(f"Loading Conversational AI Dataset ({max_samples:,} samples)...")
    
    # OpenAssistant Conversations - high quality conversational data (Parquet format, no loading scripts)
    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    
    texts = []
    for item in tqdm(dataset, desc="Conversational AI"):
        text_content = item.get('text', '').strip()
        
        # Filter for substantial conversations
        if len(text_content) > 50 and len(text_content) < 2000:
            formatted = f"Conversation:\n{text_content}"
            texts.append(formatted)
        
        if len(texts) >= max_samples:
            break
    
    print(f"Loaded {len(texts):,} conversational dialogues")
    return texts


def load_art_text(max_samples: int = 30000) -> List[str]:
    """
    Load art understanding text - TEXT ONLY (no images!)
    
    PURPOSE: Teach LLM to recognize artist names, styles, and movements from TEXT
    This creates a "conversational art expert" that can:
    - Identify: "This is Impressionism by Monet"
    - Explain: "Cubism uses geometric shapes and multiple perspectives"
    - Recommend: "For abstract art, try Kandinsky or Mondrian"
    
    NOTE: LLM learns from TEXT descriptions, NOT images
    For actual image recognition, you need CNN (VGG, ResNet, etc.)
    
    INTEGRATION WITH CNN:
    - LLM: Provides text-based art knowledge and recommendations
    - CNN: Does actual visual style transfer and image recognition
    - Together: Complete AI Art Assistant (text understanding + visual processing)
    """
    print(f"Loading Art Knowledge Dataset ({max_samples:,} samples)...")
    
    # Use WikiArt with text metadata - streaming to avoid download
    # We extract artist, style, genre and create educational Q&A format
    # NO IMAGES DOWNLOADED - only text metadata!
    dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
    
    texts = []
    count = 0
    
    for item in tqdm(dataset, desc="Art Knowledge", total=max_samples):
        # Extract text metadata (ignore the image)
        # Fields may be integers (category IDs) or strings
        artist = str(item.get('artist', '')).strip()
        style = str(item.get('style', '')).strip()
        genre = str(item.get('genre', '')).strip()
        
        if artist and style and genre and artist != '0' and style != '0':
            # Create Q&A format for art identification and explanation
            qa_texts = [
                f"Question: What artistic style is this?\nAnswer: This artwork is in the {style} style, created by {artist}. {style} is characterized by specific techniques and visual approaches typical of this movement.",
                
                f"Question: Who is the artist of this {genre} artwork?\nAnswer: This {genre} piece was created by {artist}, a prominent artist associated with the {style} movement.",
                
                f"Question: Describe the characteristics of {style} art.\nAnswer: {style} is an artistic movement represented by artists like {artist}. This style is often seen in {genre} works and has distinct visual characteristics.",
                
                f"Art Analysis:\nArtist: {artist}\nStyle: {style}\nGenre: {genre}\n\nThis artwork exemplifies {style}, a movement where {artist} made significant contributions. The {genre} genre within {style} showcases unique artistic techniques."
            ]
            
            # Add one random format to avoid repetition
            import random
            texts.append(random.choice(qa_texts))
            count += 1
        
        if count >= max_samples:
            break
    
    print(f"Loaded {len(texts):,} art knowledge samples")
    return texts


def load_c4_web(max_samples: int = 2000000) -> List[str]:
    """Load C4 (Colossal Clean Crawled Corpus) - highest quality web data"""
    print(f"Loading C4 Web Corpus ({max_samples:,} samples)...")
    
    # C4 is the cleaned web crawl used to train T5
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    texts = []
    for item in tqdm(dataset, desc="C4 Web", total=max_samples):
        text = item.get('text', '').strip()
        
        # Filter high-quality: min 200 chars, max 2000 chars
        if 200 <= len(text) <= 2000:
            texts.append(text)
        
        if len(texts) >= max_samples:
            break
    
    print(f"Loaded {len(texts):,} C4 web samples")
    return texts


def load_redpajama(max_samples: int = 1000000) -> List[str]:
    """Load RedPajama Common Crawl - high-quality diverse web data"""
    print(f"Loading RedPajama Dataset ({max_samples:,} samples)...")
    
    # RedPajama is a reproduction of LLaMA training data
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train", streaming=True)
    
    texts = []
    for item in tqdm(dataset, desc="RedPajama", total=max_samples):
        text = item.get('text', '').strip()
        
        # Filter: min 150 chars, diverse sources
        if len(text) >= 150:
            texts.append(text[:1500])  # Cap at 1500 chars
        
        if len(texts) >= max_samples:
            break
    
    print(f"Loaded {len(texts):,} RedPajama samples")
    return texts


def load_flan_instructions(max_samples: int = 500000) -> List[str]:
    """Load FLAN Collection - instruction following dataset"""
    print(f"Loading FLAN Instructions ({max_samples:,} samples)...")
    
    # FLAN is Google's instruction tuning collection
    dataset = load_dataset("Open-Orca/FLAN", split="train", streaming=True)
    
    texts = []
    for item in tqdm(dataset, desc="FLAN", total=max_samples):
        instruction = item.get('inputs', '').strip()
        response = item.get('targets', '').strip()
        
        if instruction and response:
            text = f"Instruction: {instruction}\n\nResponse: {response}"
            texts.append(text)
        
        if len(texts) >= max_samples:
            break
    
    print(f"Loaded {len(texts):,} FLAN instruction samples")
    return texts


def load_openorca(max_samples: int = 200000) -> List[str]:
    """Load OpenOrca - reasoning and explanation dataset"""
    print(f"Loading OpenOrca Reasoning ({max_samples:,} samples)...")
    
    # OpenOrca is high-quality GPT-4 explanations
    dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    
    texts = []
    for item in tqdm(dataset, desc="OpenOrca", total=max_samples):
        question = item.get('question', '').strip()
        response = item.get('response', '').strip()
        
        if question and response and len(response) > 50:
            text = f"Question: {question}\n\nAnswer: {response}"
            texts.append(text)
        
        if len(texts) >= max_samples:
            break
    
    print(f"Loaded {len(texts):,} OpenOrca reasoning samples")
    return texts


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


def load_all_datasets(squad_qa_samples: int = 200000,
                     conversational_samples: int = 500000,
                     art_text_samples: int = 100000,
                     tech_qa_samples: int = 500000,
                     c4_samples: int = 2000000,
                     redpajama_samples: int = 1000000,
                     flan_samples: int = 500000,
                     openorca_samples: int = 200000) -> List[str]:
    """
    MASSIVE-SCALE Dataset Loading for 285M Parameter Model
    
    Using SOTA datasets: C4, RedPajama, FLAN, OpenOrca
    Target: 5 MILLION samples for proper training
    
    Datasets (all streamed from HuggingFace):
    1. C4 (Colossal Clean Crawled Corpus) - 2M samples
    2. RedPajama Common Crawl - 1M samples  
    3. FLAN Collection - 500K instruction samples
    4. OpenOrca reasoning - 200K samples
    5. SQuAD 2.0 - 200K Q&A
    6. OpenAssistant/ShareGPT - 500K conversations
    7. StackOverflow/CodeSearchNet - 500K code Q&A
    8. WikiArt - 100K art descriptions
    
    Total: ~5 MILLION samples
    """
    print("=" * 70)
    print("MASSIVE-SCALE LLM TRAINING - 5 MILLION SAMPLES")
    print("=" * 70)
    print("Model: 285M params | Tokenizer: Llama 2 | GPU: RTX 3090 Ti")
    print("")
    
    all_texts = []
    
    # Load datasets in order of importance
    dataset_order = [
        ('C4 Web Corpus', c4_samples, load_c4_web),
        ('RedPajama Common Crawl', redpajama_samples, load_redpajama),
        ('FLAN Instructions', flan_samples, load_flan_instructions),
        ('OpenAssistant Conversations', conversational_samples, load_conversational_ai),
        ('StackOverflow Code Q&A', tech_qa_samples, load_ai_ml_qa),
        ('SQuAD Q&A', squad_qa_samples, load_squad_qa),
        ('OpenOrca Reasoning', openorca_samples, load_openorca),
        ('WikiArt Text', art_text_samples, load_art_text),
    ]
    
    active_datasets = [(name, count, func) for name, count, func in dataset_order if count > 0]
    dataset_count = len(active_datasets)
    
    for idx, (name, count, loader_func) in enumerate(active_datasets, 1):
        print(f"\n[{idx}/{dataset_count}] {name}")
        if loader_func == load_c4_web:
            texts = load_c4_web(max_samples=count)
        elif loader_func == load_redpajama:
            texts = load_redpajama(max_samples=count)
        elif loader_func == load_flan_instructions:
            texts = load_flan_instructions(max_samples=count)
        elif loader_func == load_openorca:
            texts = load_openorca(max_samples=count)
        elif loader_func == load_conversational_ai:
            texts = load_conversational_ai(max_samples=count)
        elif loader_func == load_ai_ml_qa:
            texts = load_ai_ml_qa(max_samples=count)
        elif loader_func == load_squad_qa:
            texts = load_squad_qa(max_samples=count)
        elif loader_func == load_art_text:
            texts = load_art_text(max_samples=count)
        else:
            continue
            
        all_texts.extend(texts)
        print(f"   Loaded {len(texts):,} samples | Total: {len(all_texts):,}")
    
    print("\n" + "=" * 70)
    print(f"TOTAL DATASET SIZE: {len(all_texts):,} samples")
    print("=" * 70)
    
    return all_texts
