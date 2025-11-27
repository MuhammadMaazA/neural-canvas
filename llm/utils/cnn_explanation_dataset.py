"""
CNN Output → Explanation Dataset
==================================
Creates training data where LLM learns to explain CNN classification results

Format:
- Input: CNN outputs (logits dict: {task: [class, confidence]})
- Output: Natural language explanation of what CNN detected and why
"""

import random
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset


# WikiArt metadata (matches your friend's CNN training data)
ARTISTS = [
    "Vincent van Gogh", "Pablo Picasso", "Claude Monet", "Leonardo da Vinci",
    "Rembrandt", "Pierre-Auguste Renoir", "Paul Cézanne", "Edgar Degas",
    "Henri Matisse", "Edvard Munch", "Wassily Kandinsky", "Gustav Klimt",
    "Salvador Dalí", "Frida Kahlo", "Johannes Vermeer", "Michelangelo",
    "Raphael", "Titian", "El Greco", "Diego Velázquez", "Francisco Goya",
    "J.M.W. Turner", "John Constable", "Eugène Delacroix", "Camille Pissarro"
]

STYLES = [
    "Impressionism", "Post-Impressionism", "Expressionism", "Cubism",
    "Abstract", "Realism", "Romanticism", "Baroque", "Renaissance",
    "Surrealism", "Pop Art", "Minimalism", "Art Nouveau", "Symbolism",
    "Fauvism", "Rococo", "Neoclassicism", "Mannerism", "Abstract Expressionism",
    "Modernism", "Contemporary", "Gothic", "Byzantine", "Pointillism",
    "Color Field Painting", "Ukiyo-e", "Northern Renaissance"
]

GENRES = [
    "Portrait", "Landscape", "Still Life", "Religious Painting", "Genre Painting",
    "Abstract", "Cityscape", "Sketch and Study", "Nude Painting", "Animal Painting",
    "History Painting", "Self-Portrait", "Mythological Painting", "Allegory",
    "Interior", "Marina", "Flower Painting", "Battle Scene", "Literary Painting"
]

# Visual features that CNNs detect for each style
STYLE_FEATURES = {
    "Impressionism": ["visible brushstrokes", "emphasis on light and its changing qualities", "ordinary subject matter", "bright colors", "loose brushwork"],
    "Post-Impressionism": ["vivid colors", "thick application of paint", "distinctive brush strokes", "real-life subject matter", "geometric forms"],
    "Expressionism": ["distorted forms", "exaggerated colors", "emotional intensity", "subjective perspective", "bold brushwork"],
    "Cubism": ["geometric shapes", "fragmented objects", "multiple viewpoints", "abstract forms", "angular compositions"],
    "Abstract": ["non-representational forms", "emphasis on color and shape", "lack of recognizable subject", "geometric or organic patterns"],
    "Realism": ["accurate detail", "photographic quality", "everyday subjects", "natural lighting", "precise brushwork"],
    "Romanticism": ["dramatic lighting", "emotional intensity", "idealized nature", "movement and action", "vivid colors"],
    "Baroque": ["dramatic contrast", "rich color", "intense light and shadow", "movement and energy", "ornate details"],
    "Renaissance": ["realistic proportions", "linear perspective", "balanced composition", "naturalistic details", "harmonious colors"],
    "Surrealism": ["dreamlike imagery", "unexpected juxtapositions", "symbolic elements", "illogical scenes", "precise technique with impossible subjects"]
}

# Genre visual characteristics
GENRE_FEATURES = {
    "Portrait": ["human subject focus", "facial features emphasized", "subject positioned centrally or prominently"],
    "Landscape": ["natural scenery", "outdoor setting", "horizon line", "sky and terrain features"],
    "Still Life": ["inanimate objects", "arranged composition", "indoor setting", "objects on table or surface"],
    "Religious Painting": ["religious figures or symbols", "sacred themes", "often formal composition"],
    "Abstract": ["non-representational", "focus on form and color", "no clear subject"],
    "Cityscape": ["urban architecture", "buildings and streets", "man-made structures"],
}


def generate_cnn_explanation_sample() -> Tuple[str, str]:
    """
    Generate one training sample: CNN output → natural explanation

    Returns:
        (input_text, target_text)
    """
    # Simulate CNN output
    artist = random.choice(ARTISTS)
    style = random.choice(STYLES)
    genre = random.choice(GENRES)

    # Simulate confidence scores (realistic distribution)
    artist_conf = random.uniform(0.65, 0.98)
    style_conf = random.uniform(0.70, 0.96)
    genre_conf = random.uniform(0.60, 0.95)

    # Format CNN output as input text (this is what the LLM will receive)
    input_formats = [
        f"CNN Classification Results:\nArtist: {artist} ({artist_conf:.1%} confidence)\nStyle: {style} ({style_conf:.1%} confidence)\nGenre: {genre} ({genre_conf:.1%} confidence)\n\nExplain these results:",

        f"The neural network classified this artwork as:\n- Artist: {artist} (confidence: {artist_conf:.2f})\n- Style: {style} (confidence: {style_conf:.2f})\n- Genre: {genre} (confidence: {genre_conf:.2f})\n\nProvide analysis:",

        f"Classification Output:\nDetected artist: {artist} ({artist_conf*100:.1f}%)\nDetected style: {style} ({style_conf*100:.1f}%)\nDetected genre: {genre} ({genre_conf*100:.1f}%)\n\nExplanation:",
    ]

    input_text = random.choice(input_formats)

    # Generate natural language explanation (this is what the LLM should learn to output)
    style_features = STYLE_FEATURES.get(style, ["characteristic visual elements"])
    genre_features = GENRE_FEATURES.get(genre, ["typical subject matter"])

    # Build explanation components
    confidence_level = "high" if min(artist_conf, style_conf, genre_conf) > 0.85 else "moderate" if min(artist_conf, style_conf, genre_conf) > 0.70 else "reasonable"

    explanations = [
        # Template 1: Technical + Art History
        f"The neural network identified this artwork with {confidence_level} confidence. The classification as {style} is based on detected visual patterns including {random.choice(style_features)} and {random.choice(style_features)}. The {genre} genre classification comes from recognizing {random.choice(genre_features)}. The attribution to {artist}, who was a prominent {style} artist known for creating {genre} works, aligns with these detected features. The CNN's confidence scores ({artist_conf:.1%} for artist, {style_conf:.1%} for style) suggest strong pattern matches in its trained feature representations.",

        # Template 2: Accessible explanation
        f"This painting was classified as {style} because the AI detected {random.choice(style_features)} - a hallmark of this movement. The {genre} genre is evident from {random.choice(genre_features)} that the convolutional layers identified. {artist} is recognized as a master of {style}, and the network's {artist_conf:.1%} confidence in this attribution suggests the visual features closely match patterns learned from this artist's known works. The model achieves these classifications by analyzing thousands of visual features across multiple layers, from basic edges and textures to complex artistic signatures.",

        # Template 3: Educational focus
        f"Let me explain how the AI reached these conclusions. First, the style classification of {style}: the neural network's early layers detected {random.choice(style_features)}, which are characteristic visual markers of {style} art. For the {genre} genre with {genre_conf:.1%} confidence, the model recognized {random.choice(genre_features)} in its mid-level feature representations. The artist attribution to {artist} at {artist_conf:.1%} confidence comes from higher-level pattern matching - the AI learned distinctive visual signatures from {artist}'s known {style} works. These confidence scores reflect how closely the detected patterns match the model's learned representations from thousands of training examples.",

        # Template 4: Conversational
        f"The AI is pretty confident about these classifications. It identified the style as {style} (confidence: {style_conf:.1%}) by detecting visual patterns like {random.choice(style_features)} - these are signature characteristics that the model learned distinguish {style} from other art movements. The {genre} classification comes from recognizing {random.choice(genre_features)}. For the artist, the network matched the painting's features to {artist}, a well-known {style} artist who frequently worked in the {genre} genre. The {artist_conf:.1%} confidence suggests the visual features strongly align with this artist's learned style signature.",

        # Template 5: Detailed CNN explanation
        f"This classification demonstrates how convolutional neural networks analyze art hierarchically. The {style} style classification ({style_conf:.1%} confidence) results from the network detecting {random.choice(style_features)} and {random.choice(style_features)} in its convolutional feature maps. These patterns activate specific neurons trained on {style} examples. The {genre} identification ({genre_conf:.1%} confidence) comes from spatial features and compositional elements: {random.choice(genre_features)}. The artist attribution to {artist} ({artist_conf:.1%} confidence) happens in deeper layers where the network has learned artist-specific visual signatures from analyzing their body of work. Higher confidence scores indicate stronger activation patterns matching the training data.",
    ]

    target_text = random.choice(explanations)

    return input_text, target_text


class CNNExplanationDataset(Dataset):
    """Dataset for training LLM to explain CNN outputs"""

    def __init__(self, num_samples: int, tokenizer, max_len: int = 512):
        """
        Args:
            num_samples: Number of training samples to generate
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
        """
        self.num_samples = num_samples
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Pre-generate all samples (faster training)
        print(f"Generating {num_samples:,} CNN explanation samples...")
        self.samples = [generate_cnn_explanation_sample() for _ in range(num_samples)]
        print("✓ Dataset generated")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_text, target_text = self.samples[idx]

        # Combine input and target for causal LM training
        full_text = f"{input_text}\n{target_text}"

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        tokens = encoding['input_ids'].squeeze(0)
        return tokens, tokens


def load_cnn_explanation_dataset(
    num_samples: int = 100000,
    tokenizer=None,
    max_len: int = 512
) -> CNNExplanationDataset:
    """
    Load dataset for training LLM to explain CNN outputs

    Args:
        num_samples: Number of samples to generate
        tokenizer: Tokenizer instance
        max_len: Maximum sequence length

    Returns:
        CNNExplanationDataset instance
    """
    print("=" * 80)
    print("CNN EXPLANATION DATASET")
    print("=" * 80)
    print(f"Training LLM to explain CNN classification outputs")
    print(f"Samples: {num_samples:,}")
    print(f"Format: CNN outputs (artist/style/genre) → natural explanations")
    print("=" * 80)

    dataset = CNNExplanationDataset(num_samples, tokenizer, max_len)

    # Show sample
    print("\nSample training pair:")
    print("-" * 80)
    input_text, target_text = dataset.samples[0]
    print("INPUT (CNN Output):")
    print(input_text)
    print("\nTARGET (Explanation LLM should generate):")
    print(target_text)
    print("=" * 80)

    return dataset


if __name__ == "__main__":
    print("Testing CNN Explanation Dataset...\n")

    # Test generation
    for i in range(3):
        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1}")
        print('='*80)
        input_text, target_text = generate_cnn_explanation_sample()
        print("INPUT:")
        print(input_text)
        print("\nOUTPUT:")
        print(target_text)

    print("\n✓ Dataset generator working correctly!")
