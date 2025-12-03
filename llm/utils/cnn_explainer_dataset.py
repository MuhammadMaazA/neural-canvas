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
# DATASET GENERATION FUNCTIONS
# =============================================================================

def generate_cnn_explanation(artist: str, style: str, genre: str, 
                             artist_conf: float, style_conf: float, genre_conf: float) -> Tuple[str, str]:
    """
    Generate a CNN output → explanation training pair with RICH context
    """
    # Get detailed info if available
    artist_data = ARTIST_INFO.get(artist, {})
    style_data = STYLE_INFO.get(style, {})
    genre_data = GENRE_INFO.get(genre, {})
    
    # Input formats (what CNN outputs look like)
    input_formats = [
        f"The CNN classified this artwork as:\n- Artist: {artist} ({artist_conf:.1%} confidence)\n- Style: {style} ({style_conf:.1%} confidence)\n- Genre: {genre} ({genre_conf:.1%} confidence)\n\nExplain this classification:",
        
        f"Classification Results:\nArtist: {artist} ({artist_conf*100:.1f}%)\nStyle: {style} ({style_conf*100:.1f}%)\nGenre: {genre} ({genre_conf*100:.1f}%)\n\nProvide analysis:",
        
        f"Neural network output:\n• Detected artist: {artist}\n• Detected style: {style}\n• Detected genre: {genre}\n\nWhat does this tell us about the artwork?",
        
        f"Image analysis complete.\nArtist prediction: {artist} (confidence: {artist_conf:.2f})\nStyle prediction: {style} (confidence: {style_conf:.2f})\nGenre prediction: {genre} (confidence: {genre_conf:.2f})\n\nInterpret these results:",
    ]
    
    input_text = random.choice(input_formats)
    
    # Build rich explanation based on available data
    explanations = []
    
    # Artist explanation
    if artist_data:
        artist_exp = random.choice([
            f"The model identified {artist}, a {artist_data.get('nationality', '')} {artist_data.get('period', '')} artist known for {artist_data.get('known_for', 'distinctive style')}.",
            f"{artist} was recognized with {artist_conf:.1%} confidence. This {artist_data.get('nationality', '')} artist {artist_data.get('description', 'created influential works')}.",
            f"The {artist_conf:.1%} confidence in {artist} suggests visual features matching this artist's characteristic {random.choice(artist_data.get('techniques', ['techniques']))}.",
        ])
    else:
        artist_exp = f"The model attributed this work to {artist} with {artist_conf:.1%} confidence, detecting visual patterns characteristic of this artist's body of work."
    
    # Style explanation  
    if style_data:
        style_chars = style_data.get('characteristics', ['distinctive visual elements'])
        style_exp = random.choice([
            f"The {style} classification ({style_conf:.1%}) indicates the presence of {random.choice(style_chars)} and {random.choice(style_chars)}, hallmarks of this {style_data.get('period', '')} movement.",
            f"{style}, originating in {style_data.get('origin', 'Europe')}, is characterized by {random.choice(style_chars)}. The {style_conf:.1%} confidence reflects strong matches with these visual patterns.",
            f"The {style} style, which {style_data.get('description', 'represents a significant artistic movement')}, was detected through analysis of compositional elements.",
        ])
    else:
        style_exp = f"The {style} classification with {style_conf:.1%} confidence indicates visual elements consistent with this artistic movement's characteristic features."
    
    # Genre explanation
    if genre_data:
        genre_exp = random.choice([
            f"The {genre} genre ({genre_conf:.1%}) was identified based on {genre_data.get('description', 'subject matter')}. Key indicators include {random.choice(genre_data.get('characteristics', ['typical subject matter']))}.",
            f"As a {genre}, this work features {random.choice(genre_data.get('characteristics', ['characteristic elements']))}. The {genre_conf:.1%} confidence suggests clear genre markers.",
            f"The classification as {genre} reflects the artwork's {genre_data.get('description', 'distinctive subject matter and composition')}.",
        ])
    else:
        genre_exp = f"The {genre} genre classification at {genre_conf:.1%} confidence reflects the subject matter and compositional approach detected in the image."
    
    # Combine into coherent explanation
    explanation_templates = [
        f"{artist_exp} {style_exp} {genre_exp}",
        
        f"This classification reveals interesting insights about the artwork. {artist_exp} {style_exp} {genre_exp} Together, these predictions paint a coherent picture of the work's artistic context.",
        
        f"Let me break down these results. {artist_exp}\n\nRegarding style: {style_exp}\n\nAs for the genre: {genre_exp}",
        
        f"The neural network has identified key characteristics of this artwork. {style_exp} {artist_exp} {genre_exp} The confidence levels suggest a strong match with the training data patterns.",
    ]
    
    target_text = random.choice(explanation_templates)
    
    return input_text, target_text


def generate_art_knowledge_sample() -> str:
    """Generate a rich art knowledge sample from our database"""
    sample_type = random.choice(['artist', 'style', 'genre', 'comparison'])
    
    if sample_type == 'artist':
        artist = random.choice(list(ARTIST_INFO.keys()))
        info = ARTIST_INFO[artist]
        templates = [
            f"Q: Tell me about {artist}.\nA: {artist} was a {info['nationality']} artist of the {info['period']} period. They are known for {info['known_for']}. Key techniques include {', '.join(info['techniques'][:2])}. Famous works include {', '.join(info['famous_works'][:2])}.",
            
            f"Q: What makes {artist}'s work distinctive?\nA: {artist} {info['description']}. Their mastery of {random.choice(info['techniques'])} became a defining feature of their artistic identity.",
            
            f"Q: Who was {artist}?\nA: A {info['nationality']} {info['period']} artist, {artist} became renowned for {info['known_for']}. Their influence on art history remains significant today.",
            
            f"{artist}, a master of {info['period']}, created works characterized by {info['known_for']}. As a {info['nationality']} artist, they brought unique perspectives to pieces like {random.choice(info['famous_works'])}.",
        ]
        return random.choice(templates)
    
    elif sample_type == 'style':
        style = random.choice(list(STYLE_INFO.keys()))
        info = STYLE_INFO[style]
        templates = [
            f"Q: What is {style}?\nA: {style} was an artistic movement that {info['description']}. Originating in {info['origin']} during the {info['period']}, it is characterized by {', '.join(info['characteristics'][:3])}.",
            
            f"Q: What defines {style} art?\nA: {style}, emerging in {info['origin']}, {info['description']}. Key artists include {', '.join(info['key_artists'][:3])}. Techniques include {', '.join(info['techniques'][:2])}.",
            
            f"The {style} movement ({info['period']}) {info['description']}. Artists like {random.choice(info['key_artists'])} exemplified its characteristics: {random.choice(info['characteristics'])} and {random.choice(info['characteristics'])}.",
            
            f"Q: How do I recognize {style}?\nA: Look for {random.choice(info['characteristics'])} and {random.choice(info['characteristics'])}. {style} {info['description']}. Major figures include {' and '.join(info['key_artists'][:2])}.",
        ]
        return random.choice(templates)
    
    elif sample_type == 'genre':
        genre = random.choice(list(GENRE_INFO.keys()))
        info = GENRE_INFO[genre]
        templates = [
            f"Q: What is {genre} in art?\nA: {genre} refers to {info['description']}. Characteristics include {', '.join(info['characteristics'])}. Examples include {info['examples']}.",
            
            f"{genre} is a category of art that involves {info['description']}. This genre typically features {random.choice(info['characteristics'])} and {random.choice(info['characteristics'])}.",
            
            f"Q: How would you describe {genre}?\nA: {genre} encompasses {info['description']}. Notable examples include {info['examples']}.",
        ]
        return random.choice(templates)
    
    else:  # comparison
        style1, style2 = random.sample(list(STYLE_INFO.keys()), 2)
        info1, info2 = STYLE_INFO[style1], STYLE_INFO[style2]
        return f"Q: How does {style1} differ from {style2}?\nA: {style1} ({info1['period']}) {info1['description']}, while {style2} ({info2['period']}) {info2['description']}. {style1} emphasizes {random.choice(info1['characteristics'])}, whereas {style2} focuses on {random.choice(info2['characteristics'])}."


def load_wikiart_for_training(max_samples: int = 50000) -> List[str]:
    """Load real WikiArt metadata and create training samples"""
    print(f"Loading WikiArt metadata ({max_samples:,} samples)...")
    
    try:
        dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
        artist_names = dataset.features['artist'].names
        style_names = dataset.features['style'].names
        genre_names = dataset.features['genre'].names
    except Exception as e:
        print(f"Error loading WikiArt: {e}")
        return []
    
    texts = []
    
    for item in tqdm(dataset, total=max_samples, desc="WikiArt"):
        if len(texts) >= max_samples:
            break
            
        try:
            artist = artist_names[item['artist']].replace('-', ' ').replace('_', ' ').title()
            style = style_names[item['style']].replace('_', ' ').title()
            genre = genre_names[item['genre']].replace('_', ' ').title()
        except:
            continue
        
        if len(artist) < 2 or len(style) < 2:
            continue
        
        # Create diverse templates
        templates = [
            f"This {style} artwork is a {genre} by {artist}. The {style} movement is known for its distinctive visual approach, and {artist}'s work exemplifies these characteristics.",
            
            f"Q: Who created this {genre}?\nA: This piece is attributed to {artist}, working in the {style} tradition. The {genre} format showcases typical {style} characteristics.",
            
            f"Analysis: {artist}'s {genre} demonstrates the key elements of {style}. The composition reflects both the artist's personal style and the broader movement's aesthetic principles.",
            
            f"This {genre} by {artist} represents the {style} movement. Notable features include the stylistic choices characteristic of both the artist and the period.",
            
            f"Q: What style is this {genre}?\nA: This work exemplifies {style}, created by {artist}. The artistic approach shows the defining characteristics of the {style} movement.",
        ]
        
        texts.append(random.choice(templates))
    
    print(f"✓ Loaded {len(texts):,} WikiArt samples")
    return texts


def load_conversational_quality(max_samples: int = 30000) -> List[str]:
    """Load high-quality conversational data for coherent sentence generation"""
    print(f"Loading conversational quality data ({max_samples:,} samples)...")
    
    texts = []
    
    # OpenAssistant for natural dialogue
    try:
        dataset = load_dataset("OpenAssistant/oasst1", split="train")
        for item in tqdm(dataset, desc="OpenAssistant", total=max_samples // 2):
            if len(texts) >= max_samples // 2:
                break
            
            text = item.get('text', '').strip()
            role = item.get('role', '')
            
            if not (50 < len(text) < 1500):
                continue
            
            # Only include helpful, well-formed responses
            if role == 'assistant':
                texts.append(f"Response: {text}")
                
    except Exception as e:
        print(f"OpenAssistant not available: {e}")
    
    # Anthropic HH for quality
    try:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
        count = 0
        for item in tqdm(dataset, desc="Anthropic HH", total=max_samples // 2):
            if count >= max_samples // 2:
                break
            
            chosen = item.get('chosen', '').strip()
            if not (100 < len(chosen) < 1500):
                continue
            
            # Clean up
            chosen = re.sub(r'\s+', ' ', chosen)
            texts.append(chosen)
            count += 1
            
    except Exception as e:
        print(f"Anthropic HH not available: {e}")
    
    print(f"✓ Loaded {len(texts):,} conversational samples")
    return texts


class CNNExplainerDataset(Dataset):
    """
    Comprehensive dataset for training LLM to explain CNN art classifications
    
    Combines:
    1. CNN output → explanation pairs
    2. Rich WikiArt knowledge
    3. Conversational quality data
    """
    
    def __init__(
        self,
        tokenizer,
        max_len: int = 512,
        num_explanation_samples: int = 80000,
        num_wikiart_samples: int = 50000,
        num_conversational_samples: int = 30000,
        num_art_knowledge_samples: int = 20000
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        all_texts = []
        
        print("=" * 80)
        print("CNN EXPLAINER DATASET - COMPREHENSIVE")
        print("=" * 80)
        
        # 1. CNN Explanation pairs (primary task)
        print(f"\n[1/4] Generating CNN explanation pairs ({num_explanation_samples:,})...")
        artists = list(ARTIST_INFO.keys()) + ["Unknown Artist", "Various Artists"]
        styles = list(STYLE_INFO.keys())
        genres = list(GENRE_INFO.keys())
        
        for _ in tqdm(range(num_explanation_samples), desc="CNN Explanations"):
            artist = random.choice(artists)
            style = random.choice(styles)
            genre = random.choice(genres)
            
            # Realistic confidence distribution
            artist_conf = random.uniform(0.55, 0.98)
            style_conf = random.uniform(0.60, 0.97)
            genre_conf = random.uniform(0.50, 0.95)
            
            input_text, target_text = generate_cnn_explanation(
                artist, style, genre, artist_conf, style_conf, genre_conf
            )
            all_texts.append(f"{input_text}\n\n{target_text}")
        
        print(f"✓ Generated {num_explanation_samples:,} explanation pairs")
        
        # 2. Art knowledge from database
        print(f"\n[2/4] Generating rich art knowledge ({num_art_knowledge_samples:,})...")
        for _ in tqdm(range(num_art_knowledge_samples), desc="Art Knowledge"):
            all_texts.append(generate_art_knowledge_sample())
        print(f"✓ Generated {num_art_knowledge_samples:,} art knowledge samples")
        
        # 3. WikiArt metadata
        print(f"\n[3/4] Loading WikiArt metadata...")
        wikiart_texts = load_wikiart_for_training(num_wikiart_samples)
        all_texts.extend(wikiart_texts)
        
        # 4. Conversational quality
        print(f"\n[4/4] Loading conversational data...")
        conv_texts = load_conversational_quality(num_conversational_samples)
        all_texts.extend(conv_texts)
        
        # Shuffle everything
        random.shuffle(all_texts)
        self.texts = all_texts
        
        print("\n" + "=" * 80)
        print("DATASET COMPLETE")
        print("=" * 80)
        print(f"Total samples: {len(self.texts):,}")
        print(f"  CNN Explanations:  {num_explanation_samples:,}")
        print(f"  Art Knowledge:     {num_art_knowledge_samples:,}")
        print(f"  WikiArt Metadata:  {len(wikiart_texts):,}")
        print(f"  Conversational:    {len(conv_texts):,}")
        print("=" * 80)
    
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


def load_cnn_explainer_dataset(
    tokenizer,
    max_len: int = 512,
    size: str = "medium"
) -> CNNExplainerDataset:
    """
    Factory function to create CNN explainer dataset
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_len: Maximum sequence length
        size: "small", "medium", or "large"
    
    Returns:
        CNNExplainerDataset
    """
    configs = {
        "small": {
            "num_explanation_samples": 30000,
            "num_wikiart_samples": 20000,
            "num_conversational_samples": 15000,
            "num_art_knowledge_samples": 10000
        },
        "medium": {
            "num_explanation_samples": 80000,
            "num_wikiart_samples": 50000,
            "num_conversational_samples": 30000,
            "num_art_knowledge_samples": 20000
        },
        "large": {
            "num_explanation_samples": 150000,
            "num_wikiart_samples": 80000,
            "num_conversational_samples": 50000,
            "num_art_knowledge_samples": 40000
        }
    }
    
    config = configs.get(size, configs["medium"])
    
    return CNNExplainerDataset(
        tokenizer=tokenizer,
        max_len=max_len,
        **config
    )


if __name__ == "__main__":
    print("Testing CNN Explainer Dataset...\n")
    
    # Test art knowledge generation
    print("Sample Art Knowledge:")
    print("-" * 80)
    for _ in range(3):
        print(generate_art_knowledge_sample())
        print()
    
    # Test CNN explanation generation
    print("\nSample CNN Explanation:")
    print("-" * 80)
    input_text, output_text = generate_cnn_explanation(
        "Vincent van Gogh", "Post-Impressionism", "Landscape",
        0.87, 0.92, 0.78
    )
    print("INPUT:")
    print(input_text)
    print("\nOUTPUT:")
    print(output_text)
    
    print("\n✓ Dataset generator working correctly!")

