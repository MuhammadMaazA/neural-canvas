from flask import Flask, request, jsonify
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Demo data - same as frontend
DEMO_CNN_RESULTS = {
    "artist": [
        {"label": "Vincent van Gogh", "confidence": 87},
        {"label": "Paul Gauguin", "confidence": 6},
        {"label": "Claude Monet", "confidence": 4},
    ],
    "style": [
        {"label": "Post-Impressionism", "confidence": 92},
        {"label": "Expressionism", "confidence": 5},
        {"label": "Impressionism", "confidence": 2},
    ],
    "genre": [
        {"label": "Landscape", "confidence": 78},
        {"label": "Cityscape", "confidence": 15},
        {"label": "Abstract", "confidence": 4},
    ],
}

LLM_OUTPUTS = {
    "scratch": "Input classified. Artist: Van Gogh. Style: Post-Impressionism. The image contains blue swirls and yellow lights. Probability high. This matches training data index 402. Night scene detected. Brush texture: impasto. Color palette: ultramarine, chrome yellow, prussian blue.",
    "distilgpt2": "This masterpiece is undeniably a Post-Impressionist work. The neural network identified the iconic heavy brushstrokes and the turbulent, swirling sky characteristic of Van Gogh's late period. The high contrast between the deep blues and the piercing yellows suggests an emotional, rather than realistic, depiction of the landscape. The cypress tree rises like a dark flame into the night sky, while the village below rests peacefully under the cosmic dance above. This is quintessential Van Gogh — raw emotion rendered in paint.",
    "hosted": "This is Vincent van Gogh's 'The Starry Night' (1889), painted during his stay at the Saint-Paul-de-Mausole asylum in Saint-Rémy-de-Provence. The work exemplifies Post-Impressionism's departure from pure optical observation. Van Gogh employs expressive, swirling brushwork to convey psychological intensity rather than atmospheric accuracy. The dominant ultramarine and cobalt blue palette, punctuated by cadmium yellow impasto stars, creates a visual rhythm that predates Expressionism. The composition balances the vertical cypress flame against horizontal village rooftops, while the turbulent sky suggests cosmic forces beyond human comprehension. This painting represents Van Gogh's synthesis of observed reality and inner emotional truth.",
}

MOCK_OUTPUTS = {
    "scratch": {"text": "Generated text from scratch model...", "delay": 3000},
    "finetuned": {"text": "Generated text from finetuned model...", "delay": 2000},
    "hosted": {"text": "Generated text from hosted model...", "delay": 1000},
}

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze an image with CNN models"""
    data = request.json
    image_url = data.get('imageUrl', '')
    
    # Simulate processing time
    time.sleep(0.1)
    
    return jsonify({
        "artist": DEMO_CNN_RESULTS["artist"],
        "style": DEMO_CNN_RESULTS["style"],
        "genre": DEMO_CNN_RESULTS["genre"],
    })

@app.route('/api/generate-llm', methods=['POST'])
def generate_llm():
    """Generate LLM output based on model selection"""
    data = request.json
    model = data.get('model', 'distilgpt2')
    
    # Simulate processing time
    time.sleep(0.3)
    
    return jsonify({
        "output": LLM_OUTPUTS.get(model, LLM_OUTPUTS["distilgpt2"])
    })

@app.route('/api/generate-text', methods=['POST'])
def generate_text():
    """Generate text from different models"""
    data = request.json
    prompt = data.get('prompt', '')
    model = data.get('model', 'hosted')
    
    # Simulate processing time based on model
    delay = MOCK_OUTPUTS.get(model, MOCK_OUTPUTS["hosted"])["delay"] / 1000
    time.sleep(delay)
    
    return jsonify({
        "text": MOCK_OUTPUTS.get(model, MOCK_OUTPUTS["hosted"])["text"]
    })

@app.route('/api/generate-diffusion', methods=['POST'])
def generate_diffusion():
    """Generate image using diffusion model"""
    data = request.json
    prompt = data.get('prompt', '')
    
    # Simulate processing time
    time.sleep(3.2)
    
    return jsonify({
        "imageUrl": "https://images.unsplash.com/photo-1541961017774-22349e4a1262?w=400&h=400&fit=crop",
        "processingTime": 3200
    })

@app.route('/api/generate-esrgan', methods=['POST'])
def generate_esrgan():
    """Generate enhanced images using ESRGAN"""
    data = request.json
    prompt = data.get('prompt', '')
    
    # Simulate staggered generation
    results = []
    delays = [2.4, 3.2, 4.0]
    image_urls = [
        "https://images.unsplash.com/photo-1578321272176-b7bbc0679853?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1579783902614-a3fb3927b6a5?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1549289524-06cf8837ace5?w=400&h=400&fit=crop",
    ]
    
    for i, (delay, url) in enumerate(zip(delays, image_urls)):
        time.sleep(delay - (delays[i-1] if i > 0 else 0))
        results.append({
            "label": f"Output {i+1}",
            "imageUrl": url,
            "processingTime": int(delay * 1000)
        })
    
    return jsonify({"results": results})

@app.route('/api/transfer-style', methods=['POST'])
def transfer_style():
    """Neural Style Transfer"""
    data = request.json
    style_image = data.get('styleImage', '')
    content_image = data.get('contentImage', '')
    
    # Simulate processing time
    time.sleep(2.1)
    
    return jsonify({
        "imageUrl": "https://images.unsplash.com/photo-1549289524-06cf8837ace5?w=800&h=800&fit=crop",
        "processingTime": 2100
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')

