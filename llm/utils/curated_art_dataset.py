"""
Curated High-Quality Dataset Loader for Art Expert Chatbot
===========================================================

Three core datasets (coursework requirement):
1. ART KNOWLEDGE: WikiArt + Best of ArtText + Museum descriptions
2. AI LITERACY: ELI5 + TruthfulQA + Science Q&A
3. CONVERSATIONAL: OpenAssistant + Anthropic HH

Quality over quantity - ~200K carefully curated samples
Better performance than 1M+ poorly filtered samples
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
        return tokens, tokens  # Input and target are the same for LM


# ==============================================================================
# DATASET 1: ART KNOWLEDGE
# ==============================================================================

def load_wikiart_knowledge(max_samples: int = 60000) -> List[str]:
    """
    HIGH-QUALITY WikiArt knowledge
    Creates rich Q&A pairs about art styles, artists, genres
    """
    print(f"\n[1/3] Loading WikiArt Art Knowledge ({max_samples:,} samples)...")

    try:
        dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
        # Get label mappings for converting IDs to names
        artist_names = dataset.features['artist'].names
        genre_names = dataset.features['genre'].names
        style_names = dataset.features['style'].names
        print(f"✓ Dataset loaded with {len(artist_names)} artists, {len(style_names)} styles, {len(genre_names)} genres")
    except Exception as e:
        print(f"✗ Error loading WikiArt: {e}")
        return []

    texts = []
    templates = [
        # EXPANDED: 30+ diverse templates to reduce repetition

        # Direct questions - varied structure
        "Q: What defines {style} art?\nA: {style} is an artistic movement characterized by {artist}'s approach to {genre}. Artists like {artist} pioneered techniques that became hallmarks of this style, focusing on distinctive visual elements and creative expression.",

        "Q: What is {style}?\nA: {style} represents a significant period in art history where artists explored {genre} through innovative techniques. The movement emphasized new ways of seeing and representing visual subjects.",

        "Q: How would you describe {style}?\nA: {style} emerged as artists like {artist} began experimenting with {genre}, creating works that challenged conventional approaches and introduced fresh perspectives to visual art.",

        "Q: What characterizes {style} art?\nA: The {style} movement is known for its distinctive treatment of {genre}, where artists developed unique methods of composition and expression that set their work apart from earlier traditions.",

        # Artist-focused - diverse phrasings
        "Q: Tell me about {artist}.\nA: {artist} was a {style} artist known for their {genre} works. Their contributions to the {style} movement helped define the aesthetic and techniques that characterize this artistic period.",

        "Q: Who was {artist}?\nA: {artist} played an important role in the {style} movement, creating influential {genre} works that demonstrated the movement's core principles and expressive possibilities.",

        "Q: What makes {artist} important?\nA: {artist} made significant contributions to {style} through innovative {genre} compositions that explored new artistic territory and influenced subsequent generations of artists.",

        "Q: Describe {artist}'s work.\nA: {artist}'s {genre} pieces exemplify {style}, showcasing techniques and aesthetic choices that became defining features of the movement during this important period in art history.",

        # Genre exploration - natural variety
        "Q: What is {genre} in art?\nA: {genre} is a category of visual art that includes works like those created by {style} artists such as {artist}. This genre encompasses specific subject matter and compositional approaches.",

        "Q: Explain {genre} as an art form.\nA: {genre} represents a distinct type of artistic expression, explored extensively by {style} artists who brought unique perspectives and techniques to this category of work.",

        "Q: What does {genre} mean in the context of {style}?\nA: Within {style}, {genre} took on particular significance as artists like {artist} used this form to explore the movement's core aesthetic principles.",

        # Historical context - varied approaches
        "Q: When did {style} develop?\nA: The {style} movement developed as artists began exploring new approaches to {genre}. Artists like {artist} were instrumental in establishing the movement's distinctive characteristics.",

        "Q: What influenced {style}?\nA: {style} emerged from evolving artistic ideas about {genre}, with pioneers like {artist} developing techniques that would define the movement's visual language.",

        # Comparative and analytical
        "Q: How is {style} different from other movements?\nA: {style} distinguished itself through its unique treatment of {genre}, with artists like {artist} pioneering methods that emphasized innovation over traditional conventions.",

        "Q: What makes {style} unique?\nA: {style} stands out for its distinctive approach to {genre}, where artists developed techniques and aesthetic principles that created a recognizable visual identity.",

        # Conversational tones
        "If you're looking at a {genre} painting with {style} characteristics, you're likely viewing work from this important artistic period. Artists like {artist} exemplify the movement through their distinctive approach.",

        "When examining {genre} works from the {style} period, notice how artists like {artist} employed techniques that became signature elements of this movement.",

        "The {style} approach to {genre} is particularly notable. Artists such as {artist} demonstrated how the movement's principles could create compelling visual experiences.",

        # Educational - natural phrasing
        "Art History Note: The {style} movement includes {genre} works by artists such as {artist}. Understanding {style} helps us recognize how artistic styles evolved and influenced visual culture.",

        "In studying {style}, we see how artists transformed {genre} through innovative techniques. {artist} exemplifies this transformation in their influential body of work.",

        "The development of {style} marked an important shift in how artists approached {genre}. Key figures like {artist} helped establish the movement's enduring influence.",

        # Image analysis - varied formats
        "Image Analysis: This artwork shows characteristics of {style}. Key indicators include the {genre} subject matter and techniques typical of artists like {artist} who worked in this movement.",

        "Analyzing this piece: The {style} influence is evident in the treatment of {genre}, reflecting techniques pioneered by artists such as {artist}.",

        "Visual examination reveals {style} characteristics in this {genre} work, demonstrating approaches associated with artists like {artist} from this period.",

        # Technique-focused
        "Q: What techniques define {style}?\nA: {style} is characterized by specific approaches to {genre}, with artists like {artist} developing methods that became hallmarks of the movement's visual expression.",

        "Q: How did {style} artists approach {genre}?\nA: Artists working in {style}, including {artist}, brought distinctive techniques to {genre}, creating works that exemplified the movement's aesthetic principles.",

        # Legacy and influence
        "Q: Why is {style} important?\nA: {style} represents a crucial development in art history. Through {genre}, artists like {artist} introduced innovations that influenced countless subsequent works.",

        "Q: What is {artist}'s legacy in {style}?\nA: {artist}'s work in {genre} helped define {style}, establishing approaches that continue to influence how we understand this important artistic movement.",

        # Simple explanations
        "The {style} movement transformed how artists thought about {genre}. Key contributors like {artist} demonstrated new possibilities for visual expression.",

        "{artist} worked within the {style} tradition, creating {genre} pieces that showcased the movement's characteristic approaches and aesthetic values.",

        "Understanding {style} means recognizing how artists like {artist} reimagined {genre}, bringing fresh perspectives to artistic practice.",
    ]

    for item in tqdm(dataset, desc="WikiArt", total=max_samples):
        # Convert IDs to actual names using ClassLabel mappings
        try:
            artist_id = item.get('artist', 0)
            style_id = item.get('style', 0)
            genre_id = item.get('genre', 0)
            
            artist = artist_names[artist_id]
            style = style_names[style_id]
            genre = genre_names[genre_id]
            
            # Clean up names - replace underscores and hyphens with spaces, title case
            artist = artist.replace('-', ' ').replace('_', ' ').title()
            style = style.replace('_', ' ').title()
            genre = genre.replace('_', ' ').title()
            
        except (IndexError, KeyError, TypeError) as e:
            continue

        # Minimal quality filter - only skip truly invalid entries
        # Keep "Unknown Artist" and "Unknown Genre" - they're still valid art knowledge!
        if len(artist) < 2 or len(style) < 2:
            continue

        # Create varied examples
        template = random.choice(templates)
        text = template.format(artist=artist, style=style, genre=genre)
        texts.append(text)

        if len(texts) >= max_samples:
            break

    print(f"✓ Loaded {len(texts):,} WikiArt knowledge samples")
    return texts


def load_arttext_descriptions(max_samples: int = 20000) -> List[str]:
    """
    ArtText dataset - high-quality art descriptions
    Better than raw captions
    """
    print(f"\nLoading ArtText Descriptions ({max_samples:,} samples)...")

    # Create synthetic high-quality art descriptions
    # (Since ArtText might not be available, we'll use a curated approach)

    styles = ["Impressionism", "Cubism", "Surrealism", "Abstract Expressionism", "Renaissance",
              "Baroque", "Romanticism", "Realism", "Post-Impressionism", "Modernism"]

    techniques = ["bold brushstrokes", "vibrant colors", "geometric forms", "dreamlike imagery",
                  "realistic details", "dramatic lighting", "emotional expression", "precise composition"]

    elements = ["light and shadow", "color harmony", "spatial depth", "textural quality",
                "compositional balance", "visual rhythm", "atmospheric perspective"]

    texts = []
    for i in range(min(max_samples, 20000)):
        style = random.choice(styles)
        technique = random.choice(techniques)
        element = random.choice(elements)

        templates = [
            f"This {style} artwork demonstrates {technique}, with particular attention to {element}. The composition reflects the movement's emphasis on innovative visual approaches.",

            f"Analysis: This piece exemplifies {style} through its use of {technique}. The artist's mastery of {element} creates a compelling visual narrative.",

            f"Q: What makes this {style}?\nA: The {technique} and emphasis on {element} are characteristic of {style}. These elements combine to create the style's distinctive aesthetic.",
        ]

        texts.append(random.choice(templates))

    print(f"✓ Generated {len(texts):,} art description samples")
    return texts


# ==============================================================================
# DATASET 2: AI LITERACY & EDUCATION
# ==============================================================================

def load_eli5_ai_focused(max_samples: int = 40000) -> List[str]:
    """
    ELI5 (Explain Like I'm Five) - FILTERED for AI/tech topics
    Perfect for teaching AI literacy in simple terms
    """
    print(f"\n[2/3] Loading ELI5 AI Literacy ({max_samples:,} samples)...")

    try:
        dataset = load_dataset("eli5_category", split="train", streaming=True)
    except:
        # Fallback to regular ELI5
        try:
            dataset = load_dataset("eli5", split="train_asks", streaming=True)
        except:
            print("ELI5 not available")
            return []

    ai_keywords = [
        'ai', 'artificial intelligence', 'machine learning', 'neural network',
        'deep learning', 'algorithm', 'computer', 'robot', 'data',
        'model', 'training', 'prediction', 'classification', 'vision',
        'language model', 'chatbot', 'automation', 'pattern recognition'
    ]

    texts = []
    processed = 0

    for item in tqdm(dataset, desc="ELI5", total=max_samples * 3):
        processed += 1
        if processed > max_samples * 5:  # Safety limit
            break

        title = str(item.get('title', '')).strip().lower()

        # STRICT filtering for AI/tech relevance
        if not any(keyword in title for keyword in ai_keywords):
            continue

        answers = item.get('answers', {}).get('text', [])
        if not answers or len(answers) == 0:
            continue

        # Get best answer
        scores = item.get('answers', {}).get('score', [])
        if scores and len(scores) > 0:
            best_idx = scores.index(max(scores))
        else:
            best_idx = 0

        answer = answers[best_idx].strip()

        # Quality filters
        if not (100 < len(answer) < 2000):
            continue

        # Remove URLs and weird formatting
        answer = re.sub(r'http\S+', '', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()

        text = f"Q: {item['title']}\n\nA (ELI5): {answer}"
        texts.append(text)

        if len(texts) >= max_samples:
            break

    print(f"✓ Loaded {len(texts):,} ELI5 samples")
    return texts


def load_science_qa(max_samples: int = 30000) -> List[str]:
    """
    Science Q&A datasets for factual accuracy
    Helps bot give precise answers
    """
    print(f"\nLoading Science Q&A ({max_samples:,} samples)...")

    try:
        dataset = load_dataset("sciq", split="train")
    except:
        print("SciQ not available")
        return []

    texts = []
    for item in tqdm(dataset, desc="SciQ"):
        if len(texts) >= max_samples:
            break

        question = item.get('question', '').strip()
        correct_answer = item.get('correct_answer', '').strip()
        support = item.get('support', '').strip()

        if not all([question, correct_answer]):
            continue

        if len(question) < 10 or len(correct_answer) < 5:
            continue

        if support and len(support) > 20:
            text = f"Q: {question}\n\nA: {correct_answer}\n\nExplanation: {support}"
        else:
            text = f"Q: {question}\n\nA: {correct_answer}"

        texts.append(text)

    print(f"✓ Loaded {len(texts):,} Science Q&A samples")
    return texts


def load_truthfulqa(max_samples: int = 5000) -> List[str]:
    """
    TruthfulQA - helps model avoid hallucinations
    Critical for AI literacy (teaching truth vs misconceptions)
    """
    print(f"\nLoading TruthfulQA ({max_samples:,} samples)...")

    try:
        dataset = load_dataset("truthful_qa", "generation", split="validation")
    except:
        print("TruthfulQA not available")
        return []

    texts = []
    for item in tqdm(dataset, desc="TruthfulQA"):
        question = item.get('question', '').strip()
        best_answer = item.get('best_answer', '').strip()

        if not all([question, best_answer]):
            continue

        text = f"Q: {question}\n\nTruthful Answer: {best_answer}"
        texts.append(text)

        if len(texts) >= max_samples:
            break

    print(f"✓ Loaded {len(texts):,} TruthfulQA samples")
    return texts


# ==============================================================================
# DATASET 3: CONVERSATIONAL QUALITY
# ==============================================================================

def load_openassistant_conversations(max_samples: int = 50000) -> List[str]:
    """
    OpenAssistant - highest quality conversational dataset
    Human-rated helpful dialogue
    """
    print(f"\n[3/3] Loading OpenAssistant Conversations ({max_samples:,} samples)...")

    try:
        dataset = load_dataset("OpenAssistant/oasst1", split="train")
    except:
        print("OpenAssistant not available")
        return []

    texts = []
    for item in tqdm(dataset, desc="OpenAssistant"):
        if len(texts) >= max_samples:
            break

        text_content = item.get('text', '').strip()
        role = item.get('role', '')

        # Quality filters
        if not (50 < len(text_content) < 2000):
            continue

        # Prefer assistant responses (higher quality)
        if role == 'assistant':
            formatted = f"Response: {text_content}"
        else:
            formatted = f"User: {text_content}"

        texts.append(formatted)

    print(f"✓ Loaded {len(texts):,} OpenAssistant dialogues")
    return texts


def load_anthropic_hh(max_samples: int = 30000) -> List[str]:
    """
    Anthropic Human Feedback - high quality, helpful responses
    """
    print(f"\nLoading Anthropic HH ({max_samples:,} samples)...")

    try:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
    except:
        print("Anthropic HH not available")
        return []

    texts = []
    for item in tqdm(dataset, desc="Anthropic HH", total=max_samples):
        if len(texts) >= max_samples:
            break

        chosen = item.get('chosen', '').strip()

        if not (100 < len(chosen) < 2000):
            continue

        # Clean up format
        chosen = chosen.replace('Human:', '\nHuman:').replace('Assistant:', '\nAssistant:')
        texts.append(chosen)

    print(f"✓ Loaded {len(texts):,} Anthropic HH samples")
    return texts


# ==============================================================================
# MAIN LOADER
# ==============================================================================

def load_curated_art_datasets(
    art_knowledge: int = 120000,
    ai_literacy: int = 150000,
    conversational: int = 150000
) -> Tuple[List[str], Dict[str, int]]:
    """
    Load EXPANDED HIGH-QUALITY curated datasets for Art Expert chatbot

    Returns ~420K samples optimized for:
    - Art style classification explanation
    - AI literacy education
    - Natural conversational flow

    Args:
        art_knowledge: Samples for art understanding (default 120K)
        ai_literacy: Samples for AI concept explanation (default 150K)
        conversational: Samples for dialogue quality (default 150K)

    Returns:
        (all_texts, dataset_stats)
    """
    print("=" * 80)
    print("EXPANDED CURATED ART EXPERT DATASET LOADING")
    print("=" * 80)
    print("Focus: More Data + Higher Quality")
    print("Target: ~420K high-value samples (EXPANDED from 235K)")
    print("=" * 80)

    all_texts = []
    stats = {}

    # DATASET 1: ART KNOWLEDGE
    print("\n" + "=" * 80)
    print("DATASET 1: ART KNOWLEDGE")
    print("=" * 80)
    art_texts = []
    
    # Load WikiArt labels (artists, styles, genres)
    art_texts.extend(load_wikiart_knowledge(max_samples=int(art_knowledge * 0.7)))  # 70% WikiArt
    
    # Add rich, detailed art knowledge (biographies, movements, techniques)
    try:
        from .rich_art_knowledge import load_curated_art_knowledge
        art_texts.extend(load_curated_art_knowledge(max_samples=int(art_knowledge * 0.3)))  # 30% rich knowledge
    except:
        print("⚠ Rich art knowledge module not found, using WikiArt only")

    all_texts.extend(art_texts)
    stats['art_knowledge'] = len(art_texts)

    # DATASET 2: AI LITERACY (EXPANDED!)
    print("\n" + "=" * 80)
    print("DATASET 2: AI LITERACY & EDUCATION (EXPANDED)")
    print("=" * 80)
    ai_texts = []
    # EXPANDED: More diverse sources for better learning
    ai_texts.extend(load_eli5_ai_focused(max_samples=int(ai_literacy * 0.40)))      # 40% ELI5 (60K)
    ai_texts.extend(load_science_qa(max_samples=int(ai_literacy * 0.27)))           # 27% SciQ (40K)
    ai_texts.extend(load_truthfulqa(max_samples=int(ai_literacy * 0.07)))           # 7% TruthfulQA (10K)
    # TODO: Add Natural Questions (13% = 20K) when available
    # TODO: Add CoQA (13% = 20K) when available

    all_texts.extend(ai_texts)
    stats['ai_literacy'] = len(ai_texts)

    # DATASET 3: CONVERSATIONAL (EXPANDED!)
    print("\n" + "=" * 80)
    print("DATASET 3: CONVERSATIONAL QUALITY (EXPANDED)")
    print("=" * 80)
    conv_texts = []
    # EXPANDED: More conversational variety
    conv_texts.extend(load_openassistant_conversations(max_samples=int(conversational * 0.53)))  # 53% OpenAssistant (80K)
    conv_texts.extend(load_anthropic_hh(max_samples=int(conversational * 0.27)))                 # 27% Anthropic (40K)
    # TODO: Add DailyDialog (13% = 20K) when available
    # TODO: Add EmpatheticDialogues (7% = 10K) when available

    all_texts.extend(conv_texts)
    stats['conversational'] = len(conv_texts)

    # Shuffle to mix datasets
    random.shuffle(all_texts)

    # Print summary
    print("\n" + "=" * 80)
    print("DATASET LOADING COMPLETE")
    print("=" * 80)
    print(f"Total Samples: {len(all_texts):,}")
    print(f"\nBreakdown:")
    print(f"  Art Knowledge:   {stats['art_knowledge']:>7,} ({stats['art_knowledge']/len(all_texts)*100:>5.1f}%)")
    print(f"  AI Literacy:     {stats['ai_literacy']:>7,} ({stats['ai_literacy']/len(all_texts)*100:>5.1f}%)")
    print(f"  Conversational:  {stats['conversational']:>7,} ({stats['conversational']/len(all_texts)*100:>5.1f}%)")
    print("=" * 80)

    return all_texts, stats


if __name__ == "__main__":
    # Test loading
    print("Testing dataset loading...")
    texts, stats = load_curated_art_datasets(
        art_knowledge=1000,
        ai_literacy=1000,
        conversational=1000
    )
    print(f"\nSample text:\n{texts[0]}\n")
    print(f"\nStats: {stats}")
