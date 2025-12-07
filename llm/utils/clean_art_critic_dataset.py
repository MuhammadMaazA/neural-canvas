"""
CNN Explainer Dataset - Comprehensive Training Data
=====================================================
Combines:
1. Real WikiArt metadata (artists, styles, genres with rich descriptions)
2. CNN output → explanation pairs (learning to explain predictions)
3. High-quality conversational data (for coherent sentence generation)

This creates the BEST possible training data for explaining CNN art classifications.
"""

import os
os.environ['HF_HOME'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['HF_DATASETS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'

import random
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm
import re


# =============================================================================
# RICH ART KNOWLEDGE DATABASE
# =============================================================================
# This contains REAL information about artists, styles, and genres from WikiArt

ARTIST_INFO = {
    "Vincent van Gogh": {
        "period": "Post-Impressionism",
        "nationality": "Dutch",
        "known_for": "bold colors, emotional honesty, and expressive brushwork",
        "techniques": ["impasto", "vibrant color palettes", "swirling brushstrokes"],
        "famous_works": ["Starry Night", "Sunflowers", "Cafe Terrace at Night"],
        "description": "pioneered expressive use of color and brushwork that influenced modern art"
    },
    "Claude Monet": {
        "period": "Impressionism",
        "nationality": "French",
        "known_for": "capturing light and atmosphere in landscapes",
        "techniques": ["broken color", "plein air painting", "light effects"],
        "famous_works": ["Water Lilies", "Impression Sunrise", "Haystacks"],
        "description": "founder of Impressionism who revolutionized landscape painting"
    },
    "Pablo Picasso": {
        "period": "Cubism/Modernism",
        "nationality": "Spanish",
        "known_for": "co-founding Cubism and constant artistic reinvention",
        "techniques": ["geometric abstraction", "multiple perspectives", "collage"],
        "famous_works": ["Guernica", "Les Demoiselles d'Avignon", "The Weeping Woman"],
        "description": "revolutionary artist who transformed modern art through multiple periods"
    },
    "Leonardo da Vinci": {
        "period": "Renaissance",
        "nationality": "Italian",
        "known_for": "mastery of sfumato and scientific approach to art",
        "techniques": ["sfumato", "chiaroscuro", "anatomical precision"],
        "famous_works": ["Mona Lisa", "The Last Supper", "Vitruvian Man"],
        "description": "Renaissance master who combined artistic genius with scientific inquiry"
    },
    "Rembrandt": {
        "period": "Baroque",
        "nationality": "Dutch",
        "known_for": "dramatic use of light and psychological depth in portraits",
        "techniques": ["chiaroscuro", "impasto highlights", "glazing"],
        "famous_works": ["The Night Watch", "Self-Portraits", "The Anatomy Lesson"],
        "description": "master of light and shadow who captured human emotion with unprecedented depth"
    },
    "Salvador Dalí": {
        "period": "Surrealism",
        "nationality": "Spanish",
        "known_for": "dreamlike imagery and technical precision",
        "techniques": ["paranoiac-critical method", "photorealistic rendering", "symbolic imagery"],
        "famous_works": ["The Persistence of Memory", "The Elephants", "Swans Reflecting Elephants"],
        "description": "Surrealist icon known for striking and bizarre imagery"
    },
    "Frida Kahlo": {
        "period": "Surrealism/Mexican Modernism",
        "nationality": "Mexican",
        "known_for": "deeply personal self-portraits and symbolic imagery",
        "techniques": ["symbolic narrative", "folk art influences", "vivid colors"],
        "famous_works": ["The Two Fridas", "Self-Portrait with Thorn Necklace", "The Broken Column"],
        "description": "created intensely personal works exploring identity, pain, and Mexican culture"
    },
    "Gustav Klimt": {
        "period": "Art Nouveau/Symbolism",
        "nationality": "Austrian",
        "known_for": "decorative patterns and gold leaf technique",
        "techniques": ["gold leaf application", "ornamental patterns", "sensual figures"],
        "famous_works": ["The Kiss", "Portrait of Adele Bloch-Bauer", "The Tree of Life"],
        "description": "leader of the Vienna Secession known for opulent, decorative works"
    },
    "Edvard Munch": {
        "period": "Expressionism",
        "nationality": "Norwegian",
        "known_for": "intense psychological and emotional themes",
        "techniques": ["expressive color", "distorted forms", "psychological symbolism"],
        "famous_works": ["The Scream", "Madonna", "The Sick Child"],
        "description": "pioneered Expressionism with works exploring anxiety and human emotion"
    },
    "Wassily Kandinsky": {
        "period": "Abstract/Expressionism",
        "nationality": "Russian",
        "known_for": "pioneering abstract art and color theory",
        "techniques": ["geometric abstraction", "color harmony", "musical composition"],
        "famous_works": ["Composition VIII", "Yellow-Red-Blue", "Several Circles"],
        "description": "pioneer of abstract art who explored the spiritual nature of color and form"
    },
    "Henri Matisse": {
        "period": "Fauvism/Modernism",
        "nationality": "French",
        "known_for": "bold use of color and fluid draughtsmanship",
        "techniques": ["fauve color", "paper cut-outs", "decorative patterns"],
        "famous_works": ["Dance", "The Red Studio", "Blue Nude"],
        "description": "leader of Fauvism known for expressive color and elegant forms"
    },
    "Johannes Vermeer": {
        "period": "Baroque/Dutch Golden Age",
        "nationality": "Dutch",
        "known_for": "masterful treatment of light in domestic scenes",
        "techniques": ["camera obscura effects", "pointillé highlights", "luminous color"],
        "famous_works": ["Girl with a Pearl Earring", "The Milkmaid", "The Art of Painting"],
        "description": "Dutch master celebrated for luminous domestic interiors"
    },
    "J.M.W. Turner": {
        "period": "Romanticism",
        "nationality": "British",
        "known_for": "atmospheric landscapes and seascapes with dramatic light",
        "techniques": ["atmospheric perspective", "luminous washes", "abstract color"],
        "famous_works": ["The Fighting Temeraire", "Rain Steam and Speed", "Snow Storm"],
        "description": "Romantic master who anticipated Impressionism with light and atmosphere"
    },
    "Edgar Degas": {
        "period": "Impressionism",
        "nationality": "French",
        "known_for": "capturing movement, especially ballet dancers",
        "techniques": ["unusual angles", "cropped compositions", "pastel work"],
        "famous_works": ["The Dance Class", "L'Absinthe", "Ballet Rehearsal"],
        "description": "Impressionist known for dynamic compositions and studies of movement"
    },
    "Pierre-Auguste Renoir": {
        "period": "Impressionism",
        "nationality": "French",
        "known_for": "joyful scenes and sensuous female figures",
        "techniques": ["feathery brushwork", "warm flesh tones", "dappled light"],
        "famous_works": ["Bal du moulin de la Galette", "Luncheon of the Boating Party"],
        "description": "Impressionist master known for vibrant, sensuous paintings of life's pleasures"
    },
    "Paul Cézanne": {
        "period": "Post-Impressionism",
        "nationality": "French",
        "known_for": "bridging Impressionism and Cubism through geometric forms",
        "techniques": ["constructive brushstrokes", "geometric simplification", "multiple viewpoints"],
        "famous_works": ["Mont Sainte-Victoire", "The Card Players", "Still Life with Apples"],
        "description": "father of modern art who influenced Cubism through structural approach to form"
    },
    "Michelangelo": {
        "period": "Renaissance",
        "nationality": "Italian",
        "known_for": "monumental figures and mastery of human anatomy",
        "techniques": ["sculptural forms", "dramatic poses", "anatomical precision"],
        "famous_works": ["Sistine Chapel Ceiling", "David", "The Creation of Adam"],
        "description": "Renaissance genius whose powerful figures defined Western art"
    },
    "Raphael": {
        "period": "Renaissance",
        "nationality": "Italian",
        "known_for": "harmonious compositions and idealized beauty",
        "techniques": ["balanced composition", "graceful figures", "atmospheric perspective"],
        "famous_works": ["The School of Athens", "Sistine Madonna", "Transfiguration"],
        "description": "High Renaissance master known for perfect harmony and grace"
    },
    "Caravaggio": {
        "period": "Baroque",
        "nationality": "Italian",
        "known_for": "dramatic chiaroscuro and theatrical realism",
        "techniques": ["tenebrism", "dramatic lighting", "psychological intensity"],
        "famous_works": ["The Calling of Saint Matthew", "Judith Beheading Holofernes"],
        "description": "revolutionary Baroque painter who transformed the use of light and shadow"
    },
    "Diego Velázquez": {
        "period": "Baroque",
        "nationality": "Spanish",
        "known_for": "royal portraits and innovative compositions",
        "techniques": ["loose brushwork", "atmospheric effects", "complex spatial arrangements"],
        "famous_works": ["Las Meninas", "The Surrender of Breda", "Portrait of Pope Innocent X"],
        "description": "Spanish master whose innovative techniques influenced Impressionism"
    }
}

STYLE_INFO = {
    "Impressionism": {
        "period": "1860s-1880s",
        "origin": "France",
        "characteristics": ["visible brushstrokes", "emphasis on light", "ordinary subjects", "outdoor painting"],
        "key_artists": ["Monet", "Renoir", "Degas", "Pissarro"],
        "techniques": ["broken color", "plein air", "capturing fleeting moments"],
        "description": "revolutionary movement that captured light and atmosphere through loose brushwork"
    },
    "Post-Impressionism": {
        "period": "1880s-1910s",
        "origin": "France",
        "characteristics": ["vivid colors", "thick paint application", "geometric forms", "emotional expression"],
        "key_artists": ["Van Gogh", "Cézanne", "Gauguin", "Seurat"],
        "techniques": ["pointillism", "expressive brushwork", "symbolic color"],
        "description": "extended Impressionism while rejecting its limitations, emphasizing structure and emotion"
    },
    "Expressionism": {
        "period": "1905-1920s",
        "origin": "Germany",
        "characteristics": ["distorted forms", "intense colors", "emotional intensity", "subjective perspective"],
        "key_artists": ["Munch", "Kirchner", "Kandinsky", "Schiele"],
        "techniques": ["bold brushwork", "exaggerated forms", "non-naturalistic color"],
        "description": "prioritized emotional experience over physical reality through distortion and intense color"
    },
    "Cubism": {
        "period": "1907-1920s",
        "origin": "France",
        "characteristics": ["geometric shapes", "fragmented forms", "multiple viewpoints", "abstract composition"],
        "key_artists": ["Picasso", "Braque", "Léger", "Gris"],
        "techniques": ["analytical fragmentation", "synthetic collage", "geometric abstraction"],
        "description": "revolutionary movement showing multiple perspectives simultaneously through geometric forms"
    },
    "Surrealism": {
        "period": "1920s-1940s",
        "origin": "France",
        "characteristics": ["dreamlike imagery", "unexpected juxtapositions", "symbolic elements", "subconscious themes"],
        "key_artists": ["Dalí", "Magritte", "Ernst", "Miró"],
        "techniques": ["automatism", "photorealistic dreamscapes", "symbolic imagery"],
        "description": "explored the subconscious mind through dreamlike and irrational imagery"
    },
    "Renaissance": {
        "period": "14th-17th century",
        "origin": "Italy",
        "characteristics": ["realistic proportions", "linear perspective", "classical themes", "naturalistic detail"],
        "key_artists": ["Leonardo", "Michelangelo", "Raphael", "Botticelli"],
        "techniques": ["sfumato", "chiaroscuro", "mathematical perspective"],
        "description": "rebirth of classical ideals emphasizing humanism, perspective, and naturalism"
    },
    "Baroque": {
        "period": "1600-1750",
        "origin": "Italy",
        "characteristics": ["dramatic lighting", "rich color", "emotional intensity", "dynamic movement"],
        "key_artists": ["Caravaggio", "Rembrandt", "Rubens", "Velázquez"],
        "techniques": ["tenebrism", "theatrical composition", "dramatic contrast"],
        "description": "dramatic and emotional style emphasizing movement, contrast, and grandeur"
    },
    "Romanticism": {
        "period": "1780-1850",
        "origin": "Europe",
        "characteristics": ["emotion over reason", "sublime nature", "individual expression", "dramatic scenes"],
        "key_artists": ["Turner", "Delacroix", "Friedrich", "Géricault"],
        "techniques": ["atmospheric effects", "expressive brushwork", "dramatic lighting"],
        "description": "emphasized emotion, individualism, and the sublime power of nature"
    },
    "Realism": {
        "period": "1840s-1880s",
        "origin": "France",
        "characteristics": ["everyday subjects", "truthful depiction", "rejection of idealization", "social themes"],
        "key_artists": ["Courbet", "Millet", "Daumier", "Eakins"],
        "techniques": ["accurate observation", "natural lighting", "detailed rendering"],
        "description": "depicted everyday life truthfully without idealization or romanticism"
    },
    "Abstract": {
        "period": "1910s-present",
        "origin": "Europe/America",
        "characteristics": ["non-representational", "pure form and color", "geometric or organic shapes"],
        "key_artists": ["Kandinsky", "Mondrian", "Malevich", "Pollock"],
        "techniques": ["color field", "geometric composition", "gestural abstraction"],
        "description": "moved away from representation to explore pure visual elements of form and color"
    },
    "Art Nouveau": {
        "period": "1890-1910",
        "origin": "Europe",
        "characteristics": ["organic forms", "decorative lines", "natural motifs", "flowing curves"],
        "key_artists": ["Klimt", "Mucha", "Gaudi", "Beardsley"],
        "techniques": ["sinuous lines", "flat decorative patterns", "ornamental detail"],
        "description": "decorative style featuring flowing organic lines inspired by natural forms"
    },
    "Fauvism": {
        "period": "1904-1908",
        "origin": "France",
        "characteristics": ["wild brushwork", "vivid non-naturalistic colors", "simplified forms"],
        "key_artists": ["Matisse", "Derain", "Vlaminck", "Braque"],
        "techniques": ["bold color application", "expressive brushwork", "flat patterns"],
        "description": "emphasized pure, vivid color over realistic representation"
    },
    "Symbolism": {
        "period": "1880s-1910s",
        "origin": "France",
        "characteristics": ["mythological themes", "dreamlike imagery", "emotional depth", "spiritual content"],
        "key_artists": ["Moreau", "Redon", "Puvis de Chavannes", "Klimt"],
        "techniques": ["symbolic imagery", "mysterious atmosphere", "rich color"],
        "description": "used symbols and imagery to express emotional and spiritual ideas"
    },
    "Rococo": {
        "period": "1720-1780",
        "origin": "France",
        "characteristics": ["ornate decoration", "pastel colors", "playful themes", "elegant figures"],
        "key_artists": ["Watteau", "Boucher", "Fragonard", "Tiepolo"],
        "techniques": ["delicate brushwork", "soft colors", "ornamental detail"],
        "description": "elegant and ornate style featuring lighthearted aristocratic subjects"
    },
    "Neoclassicism": {
        "period": "1760-1830",
        "origin": "Europe",
        "characteristics": ["classical themes", "idealized forms", "clarity", "moral subjects"],
        "key_artists": ["David", "Ingres", "Canova", "Kauffman"],
        "techniques": ["precise drawing", "balanced composition", "classical references"],
        "description": "revival of classical Greek and Roman ideals emphasizing order and clarity"
    }
}

GENRE_INFO = {
    "Portrait": {
        "description": "artwork depicting a specific person or group",
        "characteristics": ["focus on face/figure", "captures likeness", "reveals personality"],
        "examples": "Mona Lisa, Girl with a Pearl Earring"
    },
    "Landscape": {
        "description": "artwork depicting natural scenery",
        "characteristics": ["outdoor scenes", "natural elements", "atmospheric effects"],
        "examples": "Starry Night, Water Lilies, Haystacks"
    },
    "Still Life": {
        "description": "artwork depicting inanimate objects",
        "characteristics": ["arranged objects", "symbolic meaning", "technical skill display"],
        "examples": "Sunflowers, fruit bowls, vanitas paintings"
    },
    "Religious Painting": {
        "description": "artwork with religious or spiritual themes",
        "characteristics": ["biblical scenes", "saints and holy figures", "devotional purpose"],
        "examples": "Sistine Chapel, The Last Supper"
    },
    "Genre Painting": {
        "description": "artwork depicting everyday life scenes",
        "characteristics": ["ordinary people", "daily activities", "domestic settings"],
        "examples": "The Milkmaid, peasant scenes"
    },
    "History Painting": {
        "description": "artwork depicting historical, mythological, or allegorical scenes",
        "characteristics": ["grand narratives", "heroic subjects", "moral messages"],
        "examples": "Liberty Leading the People, The Death of Socrates"
    },
    "Nude Painting": {
        "description": "artwork featuring the unclothed human figure",
        "characteristics": ["anatomical study", "classical ideals", "expressive forms"],
        "examples": "The Birth of Venus, Olympia"
    },
    "Cityscape": {
        "description": "artwork depicting urban environments",
        "characteristics": ["architecture", "city life", "urban atmosphere"],
        "examples": "Cafe Terrace at Night, Paris street scenes"
    },
    "Marina": {
        "description": "artwork depicting the sea and maritime subjects",
        "characteristics": ["ocean views", "ships", "coastal scenes"],
        "examples": "The Fighting Temeraire, seascapes"
    },
    "Self-Portrait": {
        "description": "artwork where the artist depicts themselves",
        "characteristics": ["self-examination", "artistic identity", "personal expression"],
        "examples": "Rembrandt's self-portraits, Frida Kahlo's works"
    },
    "Abstract": {
        "description": "non-representational artwork",
        "characteristics": ["pure form", "color exploration", "no recognizable subjects"],
        "examples": "Composition VIII, color field paintings"
    },
    "Mythological Painting": {
        "description": "artwork depicting myths and legends",
        "characteristics": ["classical stories", "gods and heroes", "allegorical meaning"],
        "examples": "The Birth of Venus, Perseus and Andromeda"
    },
    "Animal Painting": {
        "description": "artwork featuring animals as primary subjects",
        "characteristics": ["naturalistic animals", "movement study", "character capture"],
        "examples": "horse paintings, wildlife art"
    },
    "Flower Painting": {
        "description": "artwork focusing on flowers and botanical subjects",
        "characteristics": ["detailed flora", "color harmony", "symbolic meaning"],
        "examples": "Sunflowers, Dutch flower pieces"
    }
}


# =============================================================================
# CLEAN DATASET GENERATION FUNCTIONS (NO TECHNICAL JARGON!)
# =============================================================================

def generate_artwork_explanation(artist: str, style: str, genre: str) -> Tuple[str, str]:
    """
    Generate CNN output → explanation training pair WITHOUT technical jargon
    
    Use Case 1: Explain artist/style/genre predictions like an art historian
    NO mentions of: CNN, model, confidence, neural network, detected, classified
    """
    artist_data = ARTIST_INFO.get(artist, {})
    style_data = STYLE_INFO.get(style, {})
    genre_data = GENRE_INFO.get(genre, {})
    
    # CLEAN input formats - just ask to explain the classification
    input_formats = [
        f"Explain this artwork:\nArtist: {artist}\nStyle: {style}\nGenre: {genre}",
        f"Describe the artistic characteristics:\n• Artist: {artist}\n• Style: {style}\n• Genre: {genre}",
        f"Analyze this artwork:\nCreated by: {artist}\nMovement: {style}\nType: {genre}",
        f"Provide art historical analysis for:\nArtist: {artist}, Style: {style}, Genre: {genre}",
    ]
    
    input_text = random.choice(input_formats)
    
    # Build CLEAN explanation based on available data
    explanations = []
    
    # Artist explanation - NO "model", "detected", "confidence"
    if artist_data:
        techniques = artist_data.get('techniques', ['distinctive techniques'])
        artist_explanations = [
            f"This work exemplifies {artist}'s signature {random.choice(techniques)}.",
            f"{artist}, a {artist_data.get('nationality', '')} artist from the {artist_data.get('period', '')} period, was renowned for {artist_data.get('known_for', 'distinctive artistic vision')}.",
            f"The artist's characteristic {random.choice(techniques)} is evident throughout this piece.",
            f"{artist} {artist_data.get('description', 'created influential works that shaped art history')}.",
        ]
        artist_exp = random.choice(artist_explanations)
    else:
        artist_exp = f"{artist}'s distinctive artistic approach is evident in this composition."
    
    # Style explanation - focus on art history, NOT "model predictions"
    if style_data:
        chars = style_data.get('characteristics', ['distinctive visual elements'])
        style_explanations = [
            f"This {style} work demonstrates {random.choice(chars)} and {random.choice(chars)}, hallmarks of the {style_data.get('period', '')} movement.",
            f"{style}, which emerged in {style_data.get('origin', 'Europe')} during {style_data.get('period', 'the era')}, emphasized {random.choice(chars)}.",
            f"The {style} style {style_data.get('description', 'represents a significant artistic movement')}. This is evident in the work's visual approach.",
            f"Characteristic {style} elements include {random.choice(chars)}, which distinguish this movement from its predecessors.",
        ]
        style_exp = random.choice(style_explanations)
    else:
        style_exp = f"The {style} aesthetic is apparent in the composition and execution."
    
    # Genre explanation - describe the genre, NOT "detection results"
    if genre_data:
        chars = genre_data.get('characteristics', ['characteristic elements'])
        genre_explanations = [
            f"As a {genre}, this work {genre_data.get('description', 'depicts its subject matter')}. Key features include {random.choice(chars)}.",
            f"The {genre} genre typically features {random.choice(chars)}, as seen here.",
            f"This {genre} demonstrates {genre_data.get('description', 'typical approach')}.",
        ]
        genre_exp = random.choice(genre_explanations)
    else:
        genre_exp = f"The {genre} genre is represented through the work's subject matter and composition."
    
    # Combine into NATURAL art historical explanation
    response_templates = [
        f"{artist_exp} {style_exp} {genre_exp}",
        
        f"This {genre} exemplifies {style}'s artistic philosophy. {artist_exp} {style_exp} Together, these elements create a cohesive artistic statement.",
        
        f"{style_exp} {artist_exp} The {genre} format allowed the artist to fully explore these aesthetic priorities.",
        
        f"{artist_exp} Working within the {style} movement, the artist created this {genre} that demonstrates both personal vision and period characteristics. {style_exp}",
    ]
    
    response = random.choice(response_templates)
    
    return input_text, response


def generate_artist_question(artist: str) -> Tuple[str, str]:
    """
    Generate general artist Q&A for Use Case 2
    """
    artist_data = ARTIST_INFO.get(artist, {})
    if not artist_data:
        return None, None
    
    # Question formats
    questions = [
        f"Tell me about {artist}.",
        f"Who was {artist}?",
        f"What is {artist} known for?",
        f"Describe {artist}'s artistic style.",
        f"What techniques did {artist} use?",
        f"What are {artist}'s most famous works?",
    ]
    
    question = random.choice(questions)
    
    # Build comprehensive answer
    nationality = artist_data.get('nationality', '')
    period = artist_data.get('period', '')
    known_for = artist_data.get('known_for', 'distinctive artistic vision')
    techniques = artist_data.get('techniques', [])
    famous_works = artist_data.get('famous_works', [])
    description = artist_data.get('description', 'created influential works')
    
    answer_templates = [
        f"{artist} was a {nationality} {period} artist known for {known_for}. The artist {description}. Notable techniques include {', '.join(techniques[:3])}. Major works include {', '.join(famous_works[:3])}.",
        
        f"{artist} {description}. This {nationality} artist from the {period} period was renowned for {known_for}. Characteristic techniques included {', '.join(techniques[:2])} and {techniques[2] if len(techniques) > 2 else 'innovative approaches'}.",
        
        f"A master of {period} art, {artist} created works distinguished by {known_for}. Famous pieces like {', '.join(famous_works[:2])} showcase the artist's {', '.join(techniques[:2])}.",
    ]
    
    answer = random.choice(answer_templates)
    return question, answer


def generate_style_question(style: str) -> Tuple[str, str]:
    """
    Generate general style Q&A for Use Case 2
    """
    style_data = STYLE_INFO.get(style, {})
    if not style_data:
        return None, None
    
    questions = [
        f"What is {style}?",
        f"Describe {style}.",
        f"What are the characteristics of {style}?",
        f"When did {style} emerge?",
        f"Who were the key {style} artists?",
    ]
    
    question = random.choice(questions)
    
    period = style_data.get('period', 'a significant era')
    origin = style_data.get('origin', 'Europe')
    chars = style_data.get('characteristics', [])
    artists = style_data.get('key_artists', [])
    techniques = style_data.get('techniques', [])
    description = style_data.get('description', 'represents an important artistic movement')
    
    answer_templates = [
        f"{style} was an artistic movement that emerged in {origin} during {period}. It {description}. Key characteristics include {', '.join(chars[:3])}. Prominent artists included {', '.join(artists[:3])}.",
        
        f"Emerging in {period}, {style} {description}. Artists like {', '.join(artists[:3])} employed techniques such as {', '.join(techniques[:2])} to create works characterized by {', '.join(chars[:2])}.",
        
        f"{style} originated in {origin} during {period}, characterized by {', '.join(chars[:3])}. The movement {description}, with key practitioners including {', '.join(artists[:2])}.",
    ]
    
    answer = random.choice(answer_templates)
    return question, answer


def generate_genre_question(genre: str) -> Tuple[str, str]:
    """
    Generate general genre Q&A for Use Case 2
    """
    genre_data = GENRE_INFO.get(genre, {})
    if not genre_data:
        return None, None
    
    questions = [
        f"What is {genre}?",
        f"Describe the {genre} genre.",
        f"What are characteristics of {genre}?",
    ]
    
    question = random.choice(questions)
    
    description = genre_data.get('description', 'a significant art genre')
    chars = genre_data.get('characteristics', [])
    examples = genre_data.get('examples', 'various masterpieces')
    
    answer_templates = [
        f"{genre} is {description}. Characteristic elements include {', '.join(chars[:3])}. Famous examples include {examples}.",
        
        f"The {genre} genre features {description}. Works in this genre typically display {', '.join(chars[:2])} and {chars[2] if len(chars) > 2 else 'distinctive compositional approaches'}.",
        
        f"{genre} {description}, characterized by {', '.join(chars[:3])}. Notable examples in art history include {examples}.",
    ]
    
    answer = random.choice(answer_templates)
    return question, answer


# =============================================================================
# DATASET CLASS
# =============================================================================

class CleanArtCriticDataset(Dataset):
    """
    Clean Art Critic Dataset - NO technical ML jargon!
    
    Generates TWO types of examples:
    1. Artwork explanations (Use Case 1): Explain artist/style/genre WITHOUT mentioning CNN/confidence
    2. General art Q&A (Use Case 2): Answer questions about artists, styles, genres
    
    Mix: 50% explanations, 50% general Q&A
    """
    
    def __init__(self, tokenizer, max_len: int = 512, size: str = "medium"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Determine dataset size
        sizes = {
            "small": 5_000,
            "medium": 20_000,
            "large": 50_000
        }
        self.total_examples = sizes.get(size, 20_000)
        
        print(f"\n🎨 Generating Clean Art Critic Dataset ({size})")
        print(f"   Target size: {self.total_examples:,} examples")
        print(f"   Mix: 50% artwork explanations + 50% general Q&A")
        print(f"   NO technical jargon (CNN, confidence, model, etc.)")
        
        # Get all artists, styles, genres
        self.artists = list(ARTIST_INFO.keys())
        self.styles = list(STYLE_INFO.keys())
        self.genres = list(GENRE_INFO.keys())
        
        # Pre-generate all examples
        self.examples = []
        self._generate_dataset()
        
        print(f"✅ Generated {len(self.examples):,} training examples\n")
    
    def _generate_dataset(self):
        """Generate all training examples"""
        num_explanations = self.total_examples // 2
        num_qa = self.total_examples - num_explanations
        
        # Generate artwork explanations (Use Case 1)
        print("   Generating artwork explanations...")
        for _ in tqdm(range(num_explanations), desc="Explanations"):
            artist = random.choice(self.artists)
            style = random.choice(self.styles)
            genre = random.choice(self.genres)
            
            prompt, response = generate_artwork_explanation(artist, style, genre)
            self.examples.append((prompt, response))
        
        # Generate general Q&A (Use Case 2)
        print("   Generating general art Q&A...")
        qa_types = ['artist', 'style', 'genre']
        for _ in tqdm(range(num_qa), desc="General Q&A"):
            qa_type = random.choice(qa_types)
            
            if qa_type == 'artist':
                artist = random.choice(self.artists)
                q, a = generate_artist_question(artist)
            elif qa_type == 'style':
                style = random.choice(self.styles)
                q, a = generate_style_question(style)
            else:
                genre = random.choice(self.genres)
                q, a = generate_genre_question(genre)
            
            if q and a:
                self.examples.append((q, a))
        
        # Shuffle
        random.shuffle(self.examples)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        prompt, response = self.examples[idx]
        
        # Format as: prompt + response
        full_text = f"{prompt}\n\n{response}"
        
        # Tokenize
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze()
        
        # For causal LM, labels = input_ids
        return input_ids, input_ids.clone()


def load_clean_art_critic_dataset(tokenizer, max_len: int = 512, size: str = "medium"):
    """Load clean art critic dataset"""
    return CleanArtCriticDataset(tokenizer, max_len, size)


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_examples():
    """Test a few examples to verify quality"""
    print("\n" + "="*80)
    print("CLEAN ART CRITIC DATASET - SAMPLE EXAMPLES")
    print("="*80 + "\n")
    
    # Test artwork explanation
    print("📝 EXAMPLE 1: Artwork Explanation (Use Case 1)")
    print("-" * 80)
    prompt1, response1 = generate_artwork_explanation("Vincent van Gogh", "Post-Impressionism", "Landscape")
    print(f"PROMPT:\n{prompt1}\n")
    print(f"RESPONSE:\n{response1}\n")
    
    # Test artist question
    print("\n📝 EXAMPLE 2: Artist Question (Use Case 2)")
    print("-" * 80)
    prompt2, response2 = generate_artist_question("Pablo Picasso")
    print(f"PROMPT:\n{prompt2}\n")
    print(f"RESPONSE:\n{response2}\n")
    
    # Test style question
    print("\n📝 EXAMPLE 3: Style Question (Use Case 2)")
    print("-" * 80)
    prompt3, response3 = generate_style_question("Impressionism")
    print(f"PROMPT:\n{prompt3}\n")
    print(f"RESPONSE:\n{response3}\n")
    
    print("="*80)
    print("✅ All examples generated without technical ML jargon!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_examples()
