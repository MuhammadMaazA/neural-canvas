"""
Enhanced Art Dataset with Rich Text Knowledge
==============================================
Adds detailed, educational art content to complement WikiArt labels
"""

from typing import List
import random


def load_curated_art_knowledge(max_samples: int = 20000) -> List[str]:
    """
    High-quality, manually curated art knowledge
    Better than shallow WikiArt labels
    """
    
    texts = []
    
    # ===================================================================
    # ARTIST BIOGRAPHIES (Rich, detailed information)
    # ===================================================================
    
    artist_bios = {
        "Vincent van Gogh": """Vincent van Gogh (1853-1890) was a Dutch Post-Impressionist painter who created about 2,100 artworks in just over a decade. His work is characterized by bold colors, dramatic brushwork, and emotional honesty. Despite struggling with mental illness and poverty throughout his life, his paintings like 'Starry Night' and 'Sunflowers' are now among the most recognized and valuable works in the world. He only sold one painting during his lifetime but profoundly influenced 20th-century art.""",
        
        "Leonardo da Vinci": """Leonardo da Vinci (1452-1519) was an Italian Renaissance polymath whose areas of interest included invention, drawing, painting, sculpture, architecture, science, music, mathematics, engineering, and anatomy. He is widely considered one of the greatest painters of all time. His masterpiece 'Mona Lisa' is the most famous portrait ever painted. Da Vinci was also a visionary inventor who conceptualized flying machines, armored vehicles, and solar power centuries before they became reality.""",
        
        "Pablo Picasso": """Pablo Picasso (1881-1973) was a Spanish painter, sculptor, and printmaker who spent most of his adult life in France. He co-founded the Cubist movement and invented constructed sculpture. Picasso demonstrated extraordinary artistic talent from an early age and is known for works like 'Guernica' and 'Les Demoiselles d'Avignon'. His revolutionary approach to art made him one of the most influential artists of the 20th century.""",
        
        "Claude Monet": """Claude Monet (1840-1926) was a founder of French Impressionist painting and the most consistent practitioner of the movement's philosophy of expressing perceptions of nature. His painting 'Impression, Sunrise' gave the Impressionist movement its name. Monet's focus was on capturing light and its changing qualities, often painting the same scene multiple times to show different lighting conditions. His water lily paintings from Giverny are masterpieces of color and light.""",
        
        "Rembrandt": """Rembrandt van Rijn (1606-1669) was a Dutch painter and etcher, considered one of the greatest visual artists in history and the most important in Dutch art history. His works depict biblical, historical, and everyday scenes with unprecedented emotional depth and realism. Rembrandt was a master of chiaroscuro (light and shadow), as seen in 'The Night Watch'. He created over 600 paintings, 300 etchings, and 2,000 drawings.""",
        
        "Michelangelo": """Michelangelo Buonarroti (1475-1564) was an Italian sculptor, painter, architect, and poet of the High Renaissance. His most famous works include the sculpture 'David', the Sistine Chapel ceiling frescoes, and the design of St. Peter's Basilica dome. Michelangelo's art reflected the Renaissance humanist ideal of depicting the beauty and complexity of the human form with unprecedented skill and emotional power.""",
    }
    
    for artist, bio in artist_bios.items():
        templates = [
            f"Q: Who was {artist}?\n\nA: {bio}",
            f"Q: Tell me about {artist}.\n\nA: {bio}",
            f"Q: What makes {artist} important in art history?\n\nA: {bio}",
            f"Human: Can you explain who {artist} was?\n\nAssistant: {bio}",
        ]
        for template in templates:
            texts.append(template)
    
    # ===================================================================
    # ART MOVEMENTS (Detailed explanations)
    # ===================================================================
    
    movements = {
        "Impressionism": """Impressionism was a 19th-century art movement that originated in France around the 1860s-1870s. Impressionists sought to capture the momentary effects of light and color in everyday scenes, often painting outdoors (en plein air). They used visible, loose brushstrokes and emphasized accurate depiction of light. Key artists include Claude Monet, Pierre-Auguste Renoir, and Edgar Degas. The movement revolutionized art by shifting focus from historical and mythological subjects to contemporary life and landscapes.""",
        
        "Cubism": """Cubism was a revolutionary early 20th-century art movement pioneered by Pablo Picasso and Georges Braque around 1907-1914. It abandoned traditional perspective and instead represented objects from multiple viewpoints simultaneously, breaking them into geometric shapes and reassembling them in abstracted forms. Cubism had two main phases: Analytical Cubism (monochromatic, fragmenting objects) and Synthetic Cubism (introducing collage and brighter colors). This movement fundamentally changed how artists approached representation.""",
        
        "Renaissance": """The Renaissance was a cultural movement spanning roughly the 14th to 17th centuries, beginning in Italy and spreading across Europe. In art, it marked a revival of classical Greek and Roman principles, emphasizing realism, perspective, human anatomy, and secular subjects alongside religious themes. Renaissance artists like Leonardo da Vinci, Michelangelo, and Raphael pioneered techniques like linear perspective, sfumato, and chiaroscuro. This period transformed Western art and established standards that influenced painting for centuries.""",
        
        "Baroque": """The Baroque period (roughly 1600-1750) followed the Renaissance and was characterized by dramatic use of light and shadow, intense emotions, and grandeur. Baroque art aimed to evoke emotional responses through movement, contrast, and elaborate detail. Artists like Caravaggio mastered chiaroscuro, while Rembrandt explored psychological depth. The Catholic Church promoted Baroque art during the Counter-Reformation to inspire religious devotion, but the style also flourished in secular contexts.""",
        
        "Surrealism": """Surrealism emerged in the 1920s as an artistic and literary movement exploring the unconscious mind and dreams. Led by André Breton, Surrealists like Salvador Dalí and René Magritte created dreamlike, often disturbing imagery through unexpected juxtapositions and impossible scenarios. The movement was influenced by Freudian psychology and aimed to liberate creative potential by bypassing rational thought. Surrealist works often feature melting clocks, floating objects, and bizarre transformations.""",
        
        "Post-Impressionism": """Post-Impressionism was a predominantly French art movement that developed roughly between 1886-1905, as artists moved beyond Impressionism's focus on light and color. Post-Impressionists like Vincent van Gogh, Paul Cézanne, and Paul Gauguin emphasized geometric forms, symbolic content, and emotional expression. Unlike the unified style of Impressionism, Post-Impressionist artists pursued individual artistic visions, laying groundwork for modern art movements like Fauvism and Cubism.""",
    }
    
    for movement, description in movements.items():
        templates = [
            f"Q: What is {movement}?\n\nA: {description}",
            f"Q: Explain the {movement} art movement.\n\nA: {description}",
            f"Q: Can you tell me about {movement} in art?\n\nA: {description}",
            f"Human: What characterized {movement}?\n\nAssistant: {description}",
        ]
        for template in templates:
            texts.append(template)
    
    # ===================================================================
    # ART TECHNIQUES (Practical knowledge)
    # ===================================================================
    
    techniques = {
        "chiaroscuro": """Chiaroscuro is an Italian term meaning 'light-dark' that refers to the dramatic use of strong contrasts between light and dark in art. This technique creates a sense of volume and three-dimensionality in two-dimensional works. Caravaggio and Rembrandt were masters of chiaroscuro, using it to create dramatic, emotionally powerful scenes. The technique involves carefully controlling highlights and shadows to model forms and direct the viewer's attention to key elements of the composition.""",
        
        "impasto": """Impasto is a painting technique where paint is applied very thickly to the canvas, so thick that brush or palette knife strokes remain visible. This creates a textured, three-dimensional surface that catches light in interesting ways. Vincent van Gogh famously used impasto to create movement and energy in his paintings, with thick swirls of paint creating dynamic surfaces. The technique adds physical depth and can convey emotional intensity through the vigorous application of paint.""",
        
        "sfumato": """Sfumato, Italian for 'smoky', is a painting technique pioneered by Leonardo da Vinci where colors and tones shade gradually into one another, producing soft, imperceptible transitions without harsh lines or borders. This creates a hazy, atmospheric quality. The Mona Lisa's face is the most famous example - her mysterious expression is enhanced by the soft transitions between light and shadow. The technique requires building up very thin, translucent layers of paint to achieve the subtle gradations.""",
        
        "perspective": """Linear perspective is a mathematical system for creating the illusion of depth and space on a flat surface. Developed during the Italian Renaissance in the early 15th century by architect Filippo Brunelleschi, it uses a vanishing point on the horizon line where parallel lines appear to converge. This revolutionary technique allowed artists to create convincing three-dimensional spaces and dramatically changed Western art. Masaccio's 'Trinity' and Raphael's 'School of Athens' are masterful examples.""",
        
        "fresco": """Fresco is a painting technique executed on freshly laid wet plaster. Water-based pigments are applied to the wet plaster, and as the plaster dries, the painting becomes an integral part of the wall. This technique was widely used in the Renaissance for large-scale wall paintings. Michelangelo's Sistine Chapel ceiling is perhaps the most famous fresco. The technique requires quick, confident work since artists must complete sections while the plaster is still wet, and corrections are difficult.""",
    }
    
    for technique, explanation in techniques.items():
        templates = [
            f"Q: What is {technique}?\n\nA: {explanation}",
            f"Q: Explain the {technique} technique in art.\n\nA: {explanation}",
            f"Q: Can you describe {technique}?\n\nA: {explanation}",
        ]
        for template in templates:
            texts.append(template)
    
    # ===================================================================
    # FAMOUS ARTWORKS (Specific works with context)
    # ===================================================================
    
    artworks = {
        "Mona Lisa": """The Mona Lisa, painted by Leonardo da Vinci between 1503-1519, is arguably the most famous painting in the world. It depicts a woman with an enigmatic smile against a distant landscape. Da Vinci used sfumato technique to create soft transitions between colors and tones, giving her face an almost lifelike quality. The painting's fame grew significantly after it was stolen from the Louvre in 1911 and recovered two years later. It now hangs behind bulletproof glass in the Louvre Museum, viewed by millions annually.""",
        
        "Starry Night": """Starry Night, painted by Vincent van Gogh in June 1889, depicts a swirling night sky over a French village. Van Gogh painted it from memory during his stay at the Saint-Paul-de-Mausole asylum in Saint-Rémy-de-Provence. The painting features bold, dramatic brushstrokes and vibrant blues and yellows. The cypress tree in the foreground and the swirling sky create a sense of movement and emotional intensity. Despite being created during a period of mental turmoil, it's now one of the most recognized paintings in modern culture.""",
        
        "Guernica": """Guernica, created by Pablo Picasso in 1937, is a powerful anti-war painting depicting the bombing of the Basque town of Guernica during the Spanish Civil War. The large mural-sized canvas (11 feet tall, 25 feet wide) uses only black, white, and grey tones. It shows distorted figures of people and animals in agony, conveying the chaos and suffering of war. The painting has become a universal symbol of the tragedies of war and remains one of the most powerful political statements in modern art.""",
    }
    
    for artwork, description in artworks.items():
        templates = [
            f"Q: Tell me about {artwork}.\n\nA: {description}",
            f"Q: What is {artwork}?\n\nA: {description}",
            f"Human: Can you describe {artwork}?\n\nAssistant: {description}",
        ]
        for template in templates:
            texts.append(template)
    
    print(f"✓ Generated {len(texts):,} curated art knowledge samples")
    return texts[:max_samples]
