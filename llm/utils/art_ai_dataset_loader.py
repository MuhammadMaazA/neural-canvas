"""
Art + AI Literacy Dataset Loader
Optimized for coursework: Art recognition + AI education podcast
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
    """Tokenized dataset - NO augmentation to prevent overfitting"""

    def __init__(self, texts: List[str], tokenizer, max_len: int = 512, augment: bool = False):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = False  # Disabled for stability

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


# ============================================================================
# CORE DATASET 1: ART KNOWLEDGE
# ============================================================================

def load_wikiart_text(max_samples: int = 50000) -> List[str]:
    """
    WikiArt metadata - artists, styles, genres
    Perfect for teaching chatbot about art history
    """
    print(f"Loading WikiArt Art Knowledge ({max_samples:,} samples)...")

    dataset = load_dataset("huggan/wikiart", split="train", streaming=True)

    texts = []
    count = 0

    for item in tqdm(dataset, desc="WikiArt Knowledge", total=max_samples):
        artist = str(item.get('artist', '')).strip()
        style = str(item.get('style', '')).strip()
        genre = str(item.get('genre', '')).strip()

        if artist and style and genre and artist != '0' and style != '0':
            # Create varied Q&A formats
            qa_formats = [
                f"Question: What artistic style is this?\nAnswer: This is {style} art by {artist}. {style} is known for its distinctive visual techniques and approach to composition.",

                f"Question: Who is {artist}?\nAnswer: {artist} was an artist associated with the {style} movement, known for creating {genre} works with characteristic {style} techniques.",

                f"Question: Describe {style} art.\nAnswer: {style} is an artistic movement with distinct visual characteristics. Artists like {artist} contributed significantly to this style through their {genre} works.",

                f"Art Analysis:\nArtist: {artist}\nStyle: {style}\nGenre: {genre}\n\n{style} is exemplified by artists like {artist}, who created impactful {genre} pieces that defined the movement's aesthetic.",

                f"If I see a {genre} painting with {style} characteristics, it might be by {artist}. {style} artists use specific techniques that make their work distinctive.",
            ]

            texts.append(random.choice(qa_formats))
            count += 1

        if count >= max_samples:
            break

    print(f"Loaded {len(texts):,} art knowledge samples")
    return texts


def load_art_wikipedia(max_samples: int = 20000) -> List[str]:
    """
    Wikipedia articles about art - filtered for art topics
    Provides rich context about art history, movements, techniques
    """
    print(f"Loading Art Wikipedia ({max_samples:,} samples)...")

    dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

    art_keywords = [
        'painting', 'painter', 'artist', 'artwork', 'sculpture', 'sculptor',
        'impressionism', 'cubism', 'surrealism', 'renaissance', 'baroque',
        'modern art', 'abstract', 'expressionism', 'museum', 'gallery',
        'picasso', 'monet', 'van gogh', 'da vinci', 'rembrandt'
    ]

    texts = []
    count = 0

    for item in tqdm(dataset, desc="Art Wikipedia", total=max_samples):
        title = item['title'].lower()
        text = item['text']

        # Check if art-related
        if any(keyword in title or keyword in text[:500].lower()
               for keyword in art_keywords):

            # Extract first 2-3 paragraphs
            paragraphs = text.split('\n\n')
            content_parts = []

            for para in paragraphs[:4]:
                para = para.strip()
                if len(para) > 50 and not para.startswith('=='):
                    content_parts.append(para)
                if len(content_parts) >= 2:
                    break

            if content_parts:
                content = '\n\n'.join(content_parts)
                if 100 < len(content) < 1500:
                    formatted = f"Article: {item['title']}\n\n{content}"
                    texts.append(formatted)
                    count += 1

        if count >= max_samples:
            break

    print(f"Loaded {len(texts):,} art Wikipedia articles")
    return texts


def load_artbench_descriptions(max_samples: int = 10000) -> List[str]:
    """
    ArtBench dataset - high-quality art descriptions
    """
    print(f"Loading ArtBench Descriptions ({max_samples:,} samples)...")

    try:
        # ArtBench-10 dataset
        dataset = load_dataset("Artificio/artbench", split="train", streaming=True)

        texts = []
        for item in tqdm(dataset, desc="ArtBench", total=max_samples):
            # Extract style/category information
            label = item.get('label', '')
            category = item.get('category', '')

            if label or category:
                text = f"Art Style: {label or category}\n\nThis artwork exemplifies the {label or category} style with its characteristic visual elements and techniques."
                texts.append(text)

            if len(texts) >= max_samples:
                break

        print(f"Loaded {len(texts):,} ArtBench samples")
        return texts
    except:
        print("ArtBench not available, skipping...")
        return []


# ============================================================================
# CORE DATASET 2: AI LITERACY
# ============================================================================

def load_eli5(max_samples: int = 50000) -> List[str]:
    """
    ELI5 (Explain Like I'm Five) from Reddit
    PERFECT for AI literacy - simple explanations of complex topics
    """
    print(f"Loading ELI5 AI Literacy ({max_samples:,} samples)...")

    dataset = load_dataset("eli5", split="train_asks", streaming=True)

    texts = []
    ai_keywords = ['ai', 'artificial', 'intelligence', 'machine', 'learning',
                   'neural', 'network', 'computer', 'algorithm', 'data',
                   'robot', 'deep learning', 'model', 'train', 'predict']

    for item in tqdm(dataset, desc="ELI5", total=max_samples*3):
        title = item.get('title', '').strip().lower()

        # Filter for AI/tech related questions
        if any(keyword in title for keyword in ai_keywords) or random.random() < 0.1:
            answers = item.get('answers', {}).get('text', [])
            scores = item.get('answers', {}).get('score', [])

            if answers and len(answers) > 0:
                # Get highest scored answer
                best_idx = 0
                if scores and len(scores) > 0:
                    best_idx = scores.index(max(scores))

                answer = answers[best_idx].strip()

                if 50 < len(answer) < 1500:
                    text = f"Question: {item['title']}\n\nAnswer (ELI5): {answer}"
                    texts.append(text)

        if len(texts) >= max_samples:
            break

    print(f"Loaded {len(texts):,} ELI5 explanations")
    return texts


def load_squad_qa(max_samples: int = 40000) -> List[str]:
    """SQuAD 2.0 - reading comprehension Q&A"""
    print(f"Loading SQuAD Q&A ({max_samples:,} samples)...")

    dataset = load_dataset("squad_v2", split="train")

    texts = []
    for item in tqdm(dataset, desc="SQuAD", total=max_samples):
        question = item.get('question', '').strip()
        context = item.get('context', '').strip()
        answers = item.get('answers', {}).get('text', [])

        if question and answers and len(answers) > 0:
            answer = answers[0].strip()

            if len(answer) > 10 and len(question) > 10:
                text = f"Question: {question}\n\nAnswer: {answer}\n\nContext: {context[:300]}"
                texts.append(text)

        if len(texts) >= max_samples:
            break

    print(f"Loaded {len(texts):,} SQuAD Q&A pairs")
    return texts


def load_openassistant(max_samples: int = 60000) -> List[str]:
    """
    OpenAssistant - high quality conversational AI dataset
    Good for natural dialogue and AI explanations
    """
    print(f"Loading OpenAssistant ({max_samples:,} samples)...")

    dataset = load_dataset("OpenAssistant/oasst1", split="train")

    texts = []
    for item in tqdm(dataset, desc="OpenAssistant"):
        text_content = item.get('text', '').strip()

        if 50 < len(text_content) < 2000:
            formatted = f"Conversation:\n{text_content}"
            texts.append(formatted)

        if len(texts) >= max_samples:
            break

    print(f"Loaded {len(texts):,} OpenAssistant dialogues")
    return texts


# ============================================================================
# CORE DATASET 3: CONVERSATIONAL
# ============================================================================

def load_personachat(max_samples: int = 30000) -> List[str]:
    """
    PersonaChat - conversations with personality
    Makes chatbot more engaging for podcast
    """
    print(f"Loading PersonaChat ({max_samples:,} samples)...")

    try:
        dataset = load_dataset("bavard/personachat_truecased", split="train")

        texts = []
        for item in tqdm(dataset, desc="PersonaChat"):
            # Get conversation history
            utterances = item.get('utterances', [])
            history = item.get('history', [])

            dialogue_parts = utterances if utterances else history

            if dialogue_parts and len(dialogue_parts) >= 2:
                # Take last few turns
                recent = dialogue_parts[-4:] if len(dialogue_parts) >= 4 else dialogue_parts
                dialogue = '\n'.join([f"Speaker: {turn}" for turn in recent])
                texts.append(f"Conversation:\n{dialogue}")

            if len(texts) >= max_samples:
                break

        print(f"Loaded {len(texts):,} PersonaChat dialogues")
        return texts
    except:
        print("PersonaChat not available, using alternative...")
        return []


def load_dailydialog(max_samples: int = 30000) -> List[str]:
    """
    DailyDialog - everyday conversations
    Natural dialogue flow
    """
    print(f"Loading DailyDialog ({max_samples:,} samples)...")

    try:
        dataset = load_dataset("daily_dialog", split="train")

        texts = []
        for item in tqdm(dataset, desc="DailyDialog"):
            dialog = item.get('dialog', [])

            if dialog and len(dialog) >= 3:
                # Format as conversation
                conversation = '\n'.join([f"Turn {i+1}: {turn}"
                                         for i, turn in enumerate(dialog[:6])])
                texts.append(f"Conversation:\n{conversation}")

            if len(texts) >= max_samples:
                break

        print(f"Loaded {len(texts):,} DailyDialog conversations")
        return texts
    except:
        print("DailyDialog not available, skipping...")
        return []


# ============================================================================
# BONUS DATASETS
# ============================================================================

def load_coco_captions(max_samples: int = 20000) -> List[str]:
    """
    COCO image captions
    Helps bot describe what CNN sees
    """
    print(f"Loading COCO Captions ({max_samples:,} samples)...")

    try:
        dataset = load_dataset("HuggingFaceM4/COCO", split="train", streaming=True)

        texts = []
        for item in tqdm(dataset, desc="COCO", total=max_samples):
            sentences = item.get('sentences', {}).get('raw', [])

            if sentences:
                caption = sentences[0] if isinstance(sentences, list) else sentences
                if len(caption) > 20:
                    text = f"Image description: {caption}\n\nThis describes the visual content, composition, and key elements present in the image."
                    texts.append(text)

            if len(texts) >= max_samples:
                break

        print(f"Loaded {len(texts):,} COCO captions")
        return texts
    except:
        print("COCO not available, skipping...")
        return []


# ============================================================================
# MAIN LOADER FUNCTION
# ============================================================================

def load_art_ai_literacy_datasets(
    art_knowledge_samples: int = 80000,
    ai_literacy_samples: int = 150000,
    conversational_samples: int = 60000,
    image_captions_samples: int = 20000
) -> List[str]:
    """
    Load datasets for Art + AI Literacy chatbot

    CORE 3 DATASETS (coursework requirement):
    1. Art Knowledge: WikiArt + Wikipedia + ArtBench
    2. AI Literacy: ELI5 + SQuAD + OpenAssistant
    3. Conversational: PersonaChat + DailyDialog

    BONUS:
    4. Image Captions: COCO (for CNN integration)

    Total: ~310K samples (manageable, prevents overfitting)
    """
    print("=" * 80)
    print("ART + AI LITERACY DATASET LOADING")
    print("=" * 80)
    print("Coursework Theme: AI Literacy for Art Education")
    print("Integration: CNN (recognition) + Diffusion (generation) + LLM (explanation)")
    print("=" * 80)

    all_texts = []

    # DATASET 1: ART KNOWLEDGE
    print("\n[1/4] Loading Art Knowledge Datasets...")
    art_texts = []
    if art_knowledge_samples > 0:
        art_texts.extend(load_wikiart_text(max_samples=art_knowledge_samples // 2))
        # art_texts.extend(load_art_wikipedia(max_samples=art_knowledge_samples // 3))  # DISABLED: Wikipedia dataset deprecated
        art_texts.extend(load_artbench_descriptions(max_samples=art_knowledge_samples // 6))
    all_texts.extend(art_texts)
    print(f"✓ Art Knowledge: {len(art_texts):,} samples")

    # DATASET 2: AI LITERACY
    print("\n[2/4] Loading AI Literacy Datasets...")
    ai_texts = []
    if ai_literacy_samples > 0:
        ai_texts.extend(load_eli5(max_samples=ai_literacy_samples // 3))
        ai_texts.extend(load_squad_qa(max_samples=ai_literacy_samples // 4))
        ai_texts.extend(load_openassistant(max_samples=ai_literacy_samples // 2.5))
    all_texts.extend(ai_texts)
    print(f"✓ AI Literacy: {len(ai_texts):,} samples")

    # DATASET 3: CONVERSATIONAL
    print("\n[3/4] Loading Conversational Datasets...")
    conv_texts = []
    if conversational_samples > 0:
        conv_texts.extend(load_personachat(max_samples=conversational_samples // 2))
        conv_texts.extend(load_dailydialog(max_samples=conversational_samples // 2))
    all_texts.extend(conv_texts)
    print(f"✓ Conversational: {len(conv_texts):,} samples")

    # BONUS: IMAGE CAPTIONS
    if image_captions_samples > 0:
        print("\n[4/4] Loading Image Caption Datasets...")
        caption_texts = load_coco_captions(max_samples=image_captions_samples)
        all_texts.extend(caption_texts)
        print(f"✓ Image Captions: {len(caption_texts):,} samples")

    # Shuffle to mix datasets
    random.shuffle(all_texts)

    print("\n" + "=" * 80)
    print(f"TOTAL DATASET SIZE: {len(all_texts):,} samples")
    print("=" * 80)
    print("\nDataset Breakdown:")
    print(f"  Art Knowledge:    {len(art_texts):,} samples ({len(art_texts)/len(all_texts)*100:.1f}%)")
    print(f"  AI Literacy:      {len(ai_texts):,} samples ({len(ai_texts)/len(all_texts)*100:.1f}%)")
    print(f"  Conversational:   {len(conv_texts):,} samples ({len(conv_texts)/len(all_texts)*100:.1f}%)")
    if image_captions_samples > 0:
        print(f"  Image Captions:   {len(caption_texts):,} samples ({len(caption_texts)/len(all_texts)*100:.1f}%)")
    print("=" * 80)

    return all_texts
