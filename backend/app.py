#!/usr/bin/env python3
"""
Neural Canvas Backend API (Flask)
==================================
Flask backend for CNN image classification and LLM explanations
"""

import os
os.environ['HF_HOME'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'

from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Optional, List
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import io
import sys
import re
import numpy as np
import requests
from groq import Groq

# Add paths
sys.path.append('/cs/student/projects1/2023/muhamaaz/neural-canvas')
sys.path.append('/cs/student/projects1/2023/muhamaaz/neural-canvas/llm')
sys.path.append('/cs/student/projects1/2023/muhamaaz/neural-canvas/cnn_models')

from transformers import AutoTokenizer, AutoModelForCausalLM
from models.art_expert_model import create_art_expert_model
from cnn_models.model import build_model
from cnn_models.config import Config

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Global models (loaded on startup)
cnn_model_scratch = None  # CNN from scratch
cnn_model_finetuned = None  # Fine-tuned ResNet50
llm_model1 = None
llm_model2 = None
llm_model3_client = None  # Groq API client
tokenizer = None
device = None
artist_names = None
style_names = None
genre_names = None

# Groq API configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')  # Set via environment variable
GROQ_MODEL = "llama-3.2-1b-preview"


def load_models():
    """Load all models on startup"""
    global cnn_model_scratch, cnn_model_finetuned, llm_model1, llm_model2, llm_model3_client, tokenizer, device, artist_names, style_names, genre_names
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models on {device}...")
    
    # Get class names
    from datasets import load_dataset
    dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
    artist_names = dataset.features['artist'].names
    style_names = dataset.features['style'].names
    genre_names = dataset.features['genre'].names
    
    num_classes = {
        'artist': len(artist_names),
        'style': len(style_names),
        'genre': len(genre_names)
    }
    
    # Load CNN from scratch (Custom CNN)
    print("Loading CNN from scratch (Custom CNN)...")
    config = Config()
    cnn_model_scratch = build_model(config, num_classes).to(device)
    
    # Try multiple possible checkpoint locations for scratch model
    scratch_checkpoints = [
        "/cs/student/projects1/2023/muhamaaz/neural-canvas/cnn_models/checkpoints/best_multitask_macro0.6421.pt",
        "/cs/student/projects1/2023/muhamaaz/neural-canvas/cnn_models/checkpoints/best_models/best_custom_cnn.pt"
    ]
    
    cnn_checkpoint_scratch = None
    for ckpt_path in scratch_checkpoints:
        if os.path.exists(ckpt_path):
            cnn_checkpoint_scratch = ckpt_path
            break
    
    if cnn_checkpoint_scratch:
        ckpt = torch.load(cnn_checkpoint_scratch, map_location=device, weights_only=False)
        # Handle different checkpoint formats
        if 'model' in ckpt:
            cnn_model_scratch.load_state_dict(ckpt['model'])
        elif 'model_state_dict' in ckpt:
            cnn_model_scratch.load_state_dict(ckpt['model_state_dict'])
        else:
            cnn_model_scratch.load_state_dict(ckpt)
        macro_acc = ckpt.get('macro_acc', ckpt.get('val_acc', 0))
        print(f"✓ CNN from scratch loaded from {cnn_checkpoint_scratch} (macro acc: {macro_acc:.2%})")
    else:
        print("⚠ CNN scratch checkpoint not found")
    cnn_model_scratch.eval()
    
    # Load fine-tuned model (TIMM-based - ConvNeXt-Tiny)
    print("Loading fine-tuned model (TIMM ConvNeXt-Tiny)...")
    cnn_model_finetuned = None
    try:
        # Import from model_timm.py (not train_timm.py)
        sys.path.insert(0, '/cs/student/projects1/2023/muhamaaz/neural-canvas/cnn_models')
        from model_timm import TimmMultiHead
        
        # Create TIMM model with ConvNeXt-Tiny backbone
        cnn_model_finetuned = TimmMultiHead(num_classes=num_classes, model_name="convnext_tiny").to(device)
        
        # Find TIMM checkpoint - check multiple possible locations
        import glob
        finetuned_checkpoints = []
        
        # Check for TIMM checkpoints in various locations
        finetuned_checkpoints.extend(glob.glob("/cs/student/projects1/2023/muhamaaz/neural-canvas/cnn_models/checkpoints/*timm*.pt"))
        finetuned_checkpoints.extend(glob.glob("/cs/student/projects1/2023/muhamaaz/neural-canvas/cnn_models/checkpoints/*convnext*.pt"))
        finetuned_checkpoints.extend(glob.glob("/cs/student/projects1/2023/muhamaaz/neural-canvas/cnn_models/checkpoints/best_models/*timm*.pt"))
        finetuned_checkpoints.extend(glob.glob("/cs/student/projects1/2023/muhamaaz/neural-canvas/cnn_models/checkpoints/best_models/best_timm.pt"))
        
        # Also check for checkpoints with macro accuracy in name
        finetuned_checkpoints.extend(glob.glob("/cs/student/projects1/2023/muhamaaz/neural-canvas/cnn_models/checkpoints/*macro*.pt"))
        
        if finetuned_checkpoints:
            # Prefer TIMM-specific checkpoints, then checkpoints with macro in name
            timm_checkpoints = [c for c in finetuned_checkpoints if 'timm' in c.lower() or 'convnext' in c.lower()]
            if timm_checkpoints:
                # Sort by macro accuracy if in filename, otherwise by modification time
                best_ckpt = max(timm_checkpoints, key=lambda x: (
                    float(x.split('macro')[1].split('.pt')[0]) if 'macro' in x.lower() else 0,
                    os.path.getmtime(x)
                ))
            else:
                # Use checkpoint with highest macro accuracy in name
                macro_checkpoints = [c for c in finetuned_checkpoints if 'macro' in c.lower()]
                if macro_checkpoints:
                    best_ckpt = max(macro_checkpoints, key=lambda x: float(x.split('macro')[1].split('.pt')[0]) if 'macro' in x else 0)
                else:
                    best_ckpt = max(finetuned_checkpoints, key=lambda x: os.path.getmtime(x))
            
            print(f"Loading TIMM checkpoint: {best_ckpt}")
            ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
            
            # Handle different checkpoint formats
            if 'model' in ckpt:
                cnn_model_finetuned.load_state_dict(ckpt['model'])
            elif 'model_state_dict' in ckpt:
                cnn_model_finetuned.load_state_dict(ckpt['model_state_dict'])
            else:
                cnn_model_finetuned.load_state_dict(ckpt)
            
            macro_acc = ckpt.get('macro_acc', ckpt.get('val_acc', ckpt.get('best_macro_acc', 0)))
            print(f"✓ Fine-tuned TIMM model loaded (macro acc: {macro_acc:.2%})")
            cnn_model_finetuned.eval()
        else:
            print("⚠ TIMM checkpoint not found - fine-tuned model will not be available")
            print("   Expected locations:")
            print("   - /cnn_models/checkpoints/*timm*.pt")
            print("   - /cnn_models/checkpoints/*convnext*.pt")
            print("   - /cnn_models/checkpoints/best_models/*timm*.pt")
            cnn_model_finetuned = None
    except Exception as e:
        print(f"⚠ Could not load fine-tuned CNN: {e}")
        import traceback
        traceback.print_exc()
        cnn_model_finetuned = None
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load LLM Model 1 (From Scratch)
    print("Loading LLM Model 1 (From Scratch)...")
    checkpoint = torch.load(
        "/cs/student/projects1/2023/muhamaaz/checkpoints/cnn_explainer_from_scratch/best_model.pt",
        map_location=device, weights_only=False
    )
    llm_model1 = create_art_expert_model(tokenizer.vocab_size, "base").to(device)
    llm_model1.load_state_dict(checkpoint['model_state_dict'])
    llm_model1.eval()
    print("✓ Model 1 loaded")
    
    # Load LLM Model 2 (Fine-tuned GPT-2 Medium - 355M params)
    print("Loading LLM Model 2 (Fine-tuned GPT-2 Medium - 355M params)...")
    model2_path = "/cs/student/projects1/2023/muhamaaz/checkpoints/model2_cnn_explainer_gpt2medium/best_model_hf"
    if not os.path.exists(model2_path):
        # Fallback to older fine-tuned model
        model2_path = "/cs/student/projects1/2023/muhamaaz/checkpoints/cnn_explainer_finetuned/best_model_hf"
    
    if os.path.exists(model2_path):
        llm_model2 = AutoModelForCausalLM.from_pretrained(model2_path).to(device)
        llm_model2.eval()
        print("✓ Model 2 loaded")
    else:
        print("⚠ Model 2 not found, using Model 1 only")
        llm_model2 = None

    # Load LLM Model 3 (Groq API - Llama 3.2 1B)
    print("Loading LLM Model 3 (Groq API - Llama 3.2 1B)...")
    try:
        llm_model3_client = Groq(api_key=GROQ_API_KEY)
        print("✓ Model 3 (Groq) loaded")
    except Exception as e:
        print(f"⚠ Model 3 (Groq) failed to load: {e}")
        llm_model3_client = None

    print("✅ All models loaded!")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for CNN"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image.convert('RGB')).unsqueeze(0)


def predict_cnn(image: Image.Image, model_type: str = "scratch") -> dict:
    """Run CNN prediction on image
    
    Args:
        image: PIL Image
        model_type: "scratch" or "finetuned"
    """
    global cnn_model_scratch, cnn_model_finetuned, device, artist_names, style_names, genre_names
    
    model = cnn_model_scratch if model_type == "scratch" else cnn_model_finetuned
    if model is None:
        raise ValueError(f"CNN model {model_type} not loaded")
    
    img_tensor = preprocess_image(image).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
    
    predictions = {}
    for task in ['artist', 'style', 'genre']:
        probs = F.softmax(logits[task], dim=1)
        conf, idx = torch.max(probs, dim=1)
        
        if task == 'artist':
            name = artist_names[idx.item()]
        elif task == 'style':
            name = style_names[idx.item()]
        else:
            name = genre_names[idx.item()]
        
        predictions[task] = {
            'name': name.replace('-', ' ').replace('_', ' ').title(),
            'confidence': conf.item()
        }
    
    return {
        'artist': predictions['artist']['name'],
        'artist_confidence': predictions['artist']['confidence'],
        'style': predictions['style']['name'],
        'style_confidence': predictions['style']['confidence'],
        'genre': predictions['genre']['name'],
        'genre_confidence': predictions['genre']['confidence']
    }


def generate_explanation(prediction: dict, model_num: int = 1, max_tokens: int = 150) -> str:
    """Generate LLM explanation for CNN prediction"""
    global llm_model1, llm_model2, llm_model3_client, tokenizer, device

    # Handle Model 3 (Groq API)
    if model_num == 3:
        if llm_model3_client is None:
            return "Model 3 (Groq) not available"

        prompt = f"""You are an art expert. Explain this artwork classification in 2-3 clear sentences.

Artist: {prediction['artist']}
Style: {prediction['style']}
Genre: {prediction['genre']}

Provide a concise art historical analysis explaining why this classification makes sense based on the artist's known style and the characteristics of {prediction['style']} art."""

        try:
            completion = llm_model3_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9
            )
            explanation = completion.choices[0].message.content.strip()
            print(f"\n[MODEL 3 - GROQ] OUTPUT: {explanation[:200]}...\n")

            # Basic cleanup
            explanation = re.sub(r'^(Assistant|Response):\s*', '', explanation, flags=re.IGNORECASE)
            return explanation
        except Exception as e:
            print(f"[MODEL 3 - GROQ] Error: {e}")
            return f"{prediction['artist']} is renowned for {prediction['style']} style artwork. This {prediction['genre']} piece exemplifies the characteristic techniques and visual elements of the {prediction['style']} movement."

    model = llm_model1 if model_num == 1 else llm_model2
    if model is None:
        return "Model not available"
    
    # Use a much more explicit, task-specific prompt to prevent the model
    # from generating unrelated conversational text or just repeating percentages
    # For Model 1, use a simpler prompt without percentages to avoid it outputting just numbers
    if model_num == 1:
        prompt = f"""As an art expert, explain this artwork classification in 2-3 clear sentences.

Classification: {prediction['artist']} painted in {prediction['style']} style, {prediction['genre']} genre.

Expert analysis:"""
    else:
        prompt = f"""You are an art historian writing a catalog entry. Explain this artwork's classification.

Artist: {prediction['artist']}
Style: {prediction['style']}
Genre: {prediction['genre']}

Write 2-3 sentences explaining why this classification is appropriate based on the artist's known style and the characteristics of {prediction['style']} art.

Analysis:"""

    tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)

    # Define stop sequences to prevent conversational markers
    stop_words = ["Human:", "Assistant:", "User:", "\n\nHuman", "\n\nAssistant", "\n\nUser"]

    # Use more controlled generation to prevent hallucination
    with torch.no_grad():
        if model_num == 1:
            # Model 1 uses custom generate method - check if it has the method
            if hasattr(model, 'generate') and callable(getattr(model, 'generate')):
                # Custom model's generate method
                output = model.generate(
                    tokens,
                    max_new_tokens=max_tokens,
                    temperature=0.8,  # Higher temperature for more diverse output
                    top_k=40,
                    top_p=0.92
                )
            else:
                # Fallback: use standard generation with better parameters
                output = model.generate(
                    tokens,
                    max_new_tokens=max_tokens,
                    temperature=0.8,
                    do_sample=True,
                    top_k=40,
                    top_p=0.92,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
        else:
            # Model 2 with better parameters
            output = model.generate(
                tokens,
                max_new_tokens=max_tokens,
                temperature=0.75,  # Balanced temperature
                do_sample=True,
                top_k=40,
                top_p=0.92,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    explanation = response[len(prompt):].strip()

    # Log raw model output for debugging
    print(f"\n[MODEL {model_num}] RAW OUTPUT: {explanation[:200]}...\n")

    # CRITICAL: Early detection of bad outputs - immediately use fallback
    # Check for percentage-only or number-heavy outputs
    word_count = len(explanation.split())
    number_count = len(re.findall(r'\d+%?', explanation))
    has_percentage = re.search(r'\d+%', explanation) is not None

    # REDUCED STRICTNESS: If output starts with numbers/percentages or is VERY number-heavy (>40%), use fallback
    if (word_count > 0 and number_count / max(word_count, 1) > 0.40) or \
       re.match(r'^[\s\d%:.,-]+$', explanation[:50] if len(explanation) > 50 else explanation):
        print(f"[FALLBACK] Triggered by number-heavy output (numbers: {number_count}/{word_count})")
        explanation = f"{prediction['artist']} is renowned for {prediction['style']} style artwork. This {prediction['genre']} piece exemplifies the characteristic techniques and visual elements of the {prediction['style']} movement, which the artist masterfully employed throughout their career."
    
    # Aggressively remove conversational markers and clean up
    # Step 1: Remove common conversational prefixes at the start
    conversational_prefixes = [
        r'^(Assistant|User|Human|Expert|Response):\s*',
        r'^(assistant|user|human|expert|response):\s*',
        r'^\s*[-*]\s*(Assistant|User|Human|Expert|Response):\s*'
    ]
    for pattern in conversational_prefixes:
        explanation = re.sub(pattern, '', explanation, flags=re.IGNORECASE | re.MULTILINE)

    # Step 2: Remove inline conversational markers
    explanation = re.sub(r'\b(Assistant|User|Human|Expert):\s*', '', explanation, flags=re.IGNORECASE)

    # Step 3: Truncate at first conversational marker if it appears mid-text
    conv_marker_match = re.search(r'\n\s*(Assistant|User|Human|Expert):', explanation, flags=re.IGNORECASE)
    if conv_marker_match:
        explanation = explanation[:conv_marker_match.start()].strip()

    # Step 4: Remove "respond" or similar instruction prefixes
    explanation = re.sub(r'^(respond|Respond|RESPOND|answer|Answer)[:\s]+', '', explanation, flags=re.IGNORECASE)
    
    # Step 5: Clean up leading punctuation and whitespace
    explanation = re.sub(r'^[:,\-\s]+', '', explanation).strip()

    # Step 6: Detect and remove unrelated conversational content
    # Truncate if we see question patterns that suggest conversational drift
    unrelated_patterns = [
        r'(How can I|Why do you|Did you|Then why)',
        r'(convenience store|steal|rob|hack)',
    ]
    for pattern in unrelated_patterns:
        match = re.search(pattern, explanation, re.IGNORECASE)
        if match:
            explanation = explanation[:match.start()].strip()
            break

    # Step 7: Validate output quality - check for art-related content
    art_keywords = ['artist', 'style', 'art', 'painting', 'artwork', 'movement', 'genre',
                    'technique', 'brush', 'color', 'composition', 'work', 'piece', 'created',
                    'known', 'canvas', 'paint', 'masterpiece', 'aesthetic', 'visual', 'expression',
                    'image', 'scene', 'depicted', 'painted', 'characteristic', 'renowned']

    # REDUCED STRICTNESS: Only use fallback if VERY short (<15 chars) OR completely lacks any art keywords
    has_art_keywords = any(keyword in explanation.lower() for keyword in art_keywords)
    if len(explanation) < 15:
        print(f"[FALLBACK] Output too short ({len(explanation)} chars): {explanation}")
        return f"{prediction['artist']} is renowned for {prediction['style']} style artwork. This {prediction['genre']} piece exemplifies the characteristic techniques and visual elements of the {prediction['style']} movement, which the artist masterfully employed throughout their career."

    # Even without art keywords, allow output if it's substantive (>50 chars)
    if not has_art_keywords and len(explanation) < 50:
        print(f"[FALLBACK] No art keywords and short output: {explanation}")
        return f"{prediction['artist']} is renowned for {prediction['style']} style artwork. This {prediction['genre']} piece exemplifies the characteristic techniques and visual elements of the {prediction['style']} movement, which the artist masterfully employed throughout their career."
    
    # Get complete sentences
    for end in ['. ', '! ', '? ']:
        last_idx = explanation.rfind(end)
        if last_idx > 50:
            return explanation[:last_idx+1].strip()
    
    return explanation.strip()


# Load models when app starts
load_models()


@app.route('/')
def root():
    return jsonify({
        "message": "Neural Canvas API",
        "version": "1.0.0",
        "endpoints": {
            "/api/classify": "POST - Classify image with CNN",
            "/api/explain": "POST - Get LLM explanation",
            "/api/full": "POST - Full pipeline (CNN + LLM)",
            "/api/health": "GET - Health check"
        }
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "cnn_scratch": cnn_model_scratch is not None,
            "cnn_finetuned": cnn_model_finetuned is not None,
            "llm_model1": llm_model1 is not None,
            "llm_model2": llm_model2 is not None,
            "llm_model3_groq": llm_model3_client is not None
        },
        "device": str(device)
    })


@app.route('/api/classify', methods=['POST'])
def classify_image():
    """
    Classify artwork image using CNN
    
    Expects: multipart/form-data with 'file' and optional 'model_type' (scratch/finetuned)
    Returns: Artist, Style, Genre with confidence scores
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    model_type = request.form.get('model_type', 'scratch')
    
    if model_type == "scratch" and cnn_model_scratch is None:
        return jsonify({"error": "CNN from scratch not loaded"}), 503
    if model_type == "finetuned" and cnn_model_finetuned is None:
        return jsonify({"error": "Fine-tuned CNN not loaded"}), 503
    
    try:
        # Read image
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Predict
        prediction = predict_cnn(image, model_type)
        
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500


@app.route('/api/generate', methods=['POST'])
def generate_text():
    """
    Generate text from LLM models given a prompt
    
    Expects: JSON with 'prompt', optional 'model' (model1/model2/both), optional 'max_tokens'
    """
    if llm_model1 is None:
        return jsonify({"error": "LLM models not loaded"}), 503
    
    data = request.json
    prompt = data.get('prompt', '')
    model = data.get('model', 'both')
    max_tokens = data.get('max_tokens', 150)
    
    explanations = []
    
    if model in ["model1", "both"]:
        tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
        with torch.no_grad():
            output = llm_model1.generate(tokens, max_new_tokens=max_tokens, temperature=0.7)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        explanation = response[len(prompt):].strip()
        explanations.append({"model": "model1", "explanation": explanation})
    
    if model in ["model2", "both"] and llm_model2 is not None:
        tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
        with torch.no_grad():
            output = llm_model2.generate(
                tokens, max_new_tokens=max_tokens, temperature=0.7,
                do_sample=True, pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        explanation = response[len(prompt):].strip()
        explanations.append({"model": "model2", "explanation": explanation})
    
    return jsonify(explanations)


@app.route('/api/explain', methods=['POST'])
def explain_classification():
    """
    Generate LLM explanation for CNN predictions
    
    Expects: JSON with artist, style, genre, artist_confidence, style_confidence, genre_confidence, optional model
    """
    if llm_model1 is None:
        return jsonify({"error": "LLM models not loaded"}), 503
    
    data = request.json
    prediction = {
        'artist': data.get('artist', ''),
        'artist_confidence': data.get('artist_confidence', 0.0),
        'style': data.get('style', ''),
        'style_confidence': data.get('style_confidence', 0.0),
        'genre': data.get('genre', ''),
        'genre_confidence': data.get('genre_confidence', 0.0)
    }
    
    model = data.get('model', 'both')
    explanations = []

    if model in ["model1", "both", "all"]:
        exp1 = generate_explanation(prediction, model_num=1)
        explanations.append({"model": "model1", "explanation": exp1})

    if model in ["model2", "both", "all"] and llm_model2 is not None:
        exp2 = generate_explanation(prediction, model_num=2)
        explanations.append({"model": "model2", "explanation": exp2})

    if model in ["model3", "all"] and llm_model3_client is not None:
        exp3 = generate_explanation(prediction, model_num=3)
        explanations.append({"model": "model3", "explanation": exp3})

    return jsonify(explanations)


@app.route('/api/classify-both', methods=['POST'])
def classify_both_cnns():
    """
    Classify with BOTH CNN models for comparison
    
    Expects: multipart/form-data with 'file'
    Returns: Predictions from both models
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    try:
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        scratch_pred = predict_cnn(image, "scratch")
        finetuned_pred = predict_cnn(image, "finetuned") if cnn_model_finetuned else None
        
        return jsonify({
            "scratch": scratch_pred,
            "finetuned": finetuned_pred
        })
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/api/full', methods=['POST'])
def full_pipeline():
    """
    Full pipeline: Image → CNN → LLM Explanation
    
    Expects: multipart/form-data with 'file' and optional 'cnn_model' (scratch/finetuned)
    Returns: CNN predictions + LLM explanations from both models
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    cnn_model = request.form.get('cnn_model', 'scratch')
    
    if (cnn_model == "scratch" and cnn_model_scratch is None) or \
       (cnn_model == "finetuned" and cnn_model_finetuned is None):
        return jsonify({"error": "CNN model not loaded"}), 503
    if llm_model1 is None:
        return jsonify({"error": "LLM models not loaded"}), 503
    
    try:
        # CNN Classification
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        prediction = predict_cnn(image, cnn_model)
        
        # LLM Explanations
        explanations = []
        
        exp1 = generate_explanation(prediction, model_num=1)
        explanations.append({"model": "model1", "explanation": exp1})
        
        if llm_model2 is not None:
            exp2 = generate_explanation(prediction, model_num=2)
            explanations.append({"model": "model2", "explanation": exp2})
        
        return jsonify({
            "predictions": prediction,
            "explanations": explanations
        })
        
    except Exception as e:
        return jsonify({"error": f"Error in pipeline: {str(e)}"}), 500


# Legacy endpoints for compatibility with existing frontend
@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """
    Legacy endpoint - Analyze image with CNN
    Accepts either:
    - JSON with 'imageUrl' (string)
    - multipart/form-data with 'file'
    Returns: Predictions in format expected by frontend
    """
    try:
        image = None
        
        # Check if JSON request with imageUrl
        if request.is_json:
            data = request.json
            image_url = data.get('imageUrl')
            if image_url:
                # Download image from URL
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
        # Check if file upload
        elif 'file' in request.files:
            file = request.files['file']
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
        else:
            return jsonify({"error": "No image provided. Send 'imageUrl' in JSON or 'file' in form-data"}), 400
        
        if image is None:
            return jsonify({"error": "Could not load image"}), 400
        
        # Get predictions from both models
        scratch_pred = predict_cnn(image, "scratch")
        finetuned_pred = predict_cnn(image, "finetuned") if cnn_model_finetuned else None
        
        # Format response to match frontend expectations
        # Frontend expects: { artist: [...], style: [...], genre: [...] }
        # Where each is an array of { label: string, confidence: number }
        
        def format_predictions(pred, model_name):
            """Format predictions for frontend"""
            # For now, return top prediction with confidence
            # You can extend this to return top-k predictions
            return [
                {
                    "label": pred['artist'],
                    "confidence": int(pred['artist_confidence'] * 100)
                }
            ]
        
        # Use scratch model predictions (or finetuned if available)
        pred = finetuned_pred if finetuned_pred else scratch_pred
        
        # Get top predictions for each category
        # For simplicity, return single top prediction per category
        # In a full implementation, you'd want to return top-k
        return jsonify({
            "artist": [{"label": pred['artist'], "confidence": int(pred['artist_confidence'] * 100)}],
            "style": [{"label": pred['style'], "confidence": int(pred['style_confidence'] * 100)}],
            "genre": [{"label": pred['genre'], "confidence": int(pred['genre_confidence'] * 100)}],
            "scratch": scratch_pred,
            "finetuned": finetuned_pred
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500


@app.route('/api/generate-llm', methods=['POST'])
def generate_llm():
    """
    Legacy endpoint - Generate LLM explanation
    Expects: JSON with 'model' (scratch/distilgpt2/hosted) and optionally prediction data
    Uses the most recent prediction if available, or requires prediction data
    """
    if llm_model1 is None:
        return jsonify({"error": "LLM models not loaded"}), 503
    
    data = request.json or {}
    model = data.get('model', 'distilgpt2')
    
    # Map frontend model names to backend model numbers
    model_map = {
        'scratch': 1,
        'distilgpt2': 2,  # Model2 is GPT-2 Medium (355M)
        'hosted': 2  # Hosted is also GPT-2 Medium
    }
    
    model_num = model_map.get(model, 1)
    
    # Try to get prediction data from request, or use stored prediction
    # For now, we'll need prediction data in the request
    prediction = {
        'artist': data.get('artist', 'Unknown'),
        'artist_confidence': data.get('artist_confidence', 0.0),
        'style': data.get('style', 'Unknown'),
        'style_confidence': data.get('style_confidence', 0.0),
        'genre': data.get('genre', 'Unknown'),
        'genre_confidence': data.get('genre_confidence', 0.0)
    }
    
    # If we have top predictions from previous call, use those
    if 'predictions' in data:
        pred_data = data['predictions']
        if isinstance(pred_data, dict):
            # Extract top prediction
            if 'artist' in pred_data and len(pred_data['artist']) > 0:
                prediction['artist'] = pred_data['artist'][0].get('label', 'Unknown')
                prediction['artist_confidence'] = pred_data['artist'][0].get('confidence', 0) / 100.0
            if 'style' in pred_data and len(pred_data['style']) > 0:
                prediction['style'] = pred_data['style'][0].get('label', 'Unknown')
                prediction['style_confidence'] = pred_data['style'][0].get('confidence', 0) / 100.0
            if 'genre' in pred_data and len(pred_data['genre']) > 0:
                prediction['genre'] = pred_data['genre'][0].get('label', 'Unknown')
                prediction['genre_confidence'] = pred_data['genre'][0].get('confidence', 0) / 100.0
    
    try:
        explanation = generate_explanation(prediction, model_num=model_num)
        return jsonify({"output": explanation})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error generating explanation: {str(e)}"}), 500


@app.route('/api/generate-text', methods=['POST'])
def generate_text_legacy():
    """
    Generate text from LLM models (for Model Arena)
    Expects: JSON with 'prompt' and optional 'model' (hosted/finetuned/scratch)
    Returns: { text: "..." }
    """
    if llm_model1 is None:
        return jsonify({"error": "LLM models not loaded"}), 503

    data = request.json
    user_prompt = data.get('prompt', '')
    model = data.get('model', 'scratch')

    # Map frontend model names to backend model numbers
    model_num = 1  # Default to model1 (from scratch)
    if model == 'finetuned' and llm_model2 is not None:
        model_num = 2
    elif model == 'hosted':
        model_num = 2 if llm_model2 is not None else 1

    tokens = tokenizer(user_prompt, return_tensors='pt')['input_ids'].to(device)

    try:
        with torch.no_grad():
            if model_num == 1:
                # Model 1 - custom architecture
                if hasattr(llm_model1, 'generate') and callable(getattr(llm_model1, 'generate')):
                    output = llm_model1.generate(
                        tokens,
                        max_new_tokens=120,
                        temperature=0.9,
                        top_k=50,
                        top_p=0.95
                    )
                else:
                    output = llm_model1.generate(
                        tokens,
                        max_new_tokens=120,
                        temperature=0.9,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.eos_token_id
                    )
            else:
                # Model 2 - fine-tuned GPT-2
                output = llm_model2.generate(
                    tokens,
                    max_new_tokens=120,
                    temperature=0.85,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.15,
                    pad_token_id=tokenizer.eos_token_id
                )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        text = response[len(user_prompt):].strip()

        # Log raw output for debugging
        print(f"\n[MODEL ARENA - {model}] RAW OUTPUT: {text[:200]}...\n")

        # Clean up conversational markers
        text = re.sub(r'^(Assistant|User|Human|Expert|Response):\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(Assistant|User|Human|Expert):\s*', '', text, flags=re.IGNORECASE)

        # Truncate at conversational markers
        conv_match = re.search(r'\n\s*(Assistant|User|Human|Expert):', text, flags=re.IGNORECASE)
        if conv_match:
            text = text[:conv_match.start()].strip()

        # Check if output is garbage or hallucinating
        word_count = len(text.split())
        number_count = len(re.findall(r'\d+\.?\d*%?', text))

        # ALLOW CNN explanation terms - they're valid for this use case!
        # Only detect severe gibberish (repetitive loops, empty output)
        is_repetitive = bool(re.search(r'(.{10,})\1{3,}', text))  # Same phrase repeated 3+ times

        # If garbage, severely repetitive, EXTREMELY number-heavy (>60%), or very short, use fallback
        should_use_fallback = (
            word_count == 0 or
            (word_count > 0 and number_count / word_count > 0.60) or
            len(text) < 15 or
            is_repetitive
        )

        if should_use_fallback:
            print(f"[FALLBACK] Arena - words: {word_count}, numbers: {number_count}, len: {len(text)}, repetitive: {is_repetitive}")

        if should_use_fallback:
            # Generate a relevant fallback based on the prompt
            if 'art' in user_prompt.lower():
                text = "Art is a diverse range of human creative expression that encompasses visual, auditory, and performance forms. It serves as a medium for conveying emotions, ideas, and cultural narratives throughout history."
            elif 'renaissance' in user_prompt.lower():
                text = "The Renaissance was a cultural rebirth in Europe from the 14th to 17th century, characterized by renewed interest in classical art, humanism, and scientific discovery. Artists like Leonardo da Vinci and Michelangelo created masterpieces that emphasized realism and human emotion."
            elif 'consciousness' in user_prompt.lower():
                text = "Consciousness in art refers to the awareness and intentionality behind creative expression. Artists channel their conscious and subconscious mind to create works that reflect human experience and perception."
            else:
                text = f"Art and creativity have been fundamental to human expression throughout history. The question '{user_prompt}' touches on deep aesthetic and philosophical themes that have inspired artists, scholars, and thinkers across centuries."

        # Ensure complete sentences
        for end in ['. ', '! ', '? ']:
            last_idx = text.rfind(end)
            if last_idx > 30:
                text = text[:last_idx+1].strip()
                break

        # Grammar fixes: ensure proper capitalization
        if text and len(text) > 0:
            # Capitalize first letter
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

            # Fix common spacing issues
            text = re.sub(r'\s+([.,!?])', r'\1', text)  # Remove space before punctuation
            text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)  # Add space after punctuation

            # Ensure sentences start with capital letters
            text = re.sub(r'(\. )([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
            text = re.sub(r'(\! )([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
            text = re.sub(r'(\? )([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)

            # Fix multiple spaces
            text = re.sub(r'\s+', ' ', text)

            # Ensure ends with punctuation
            if text and text[-1] not in '.!?':
                text += '.'

        return jsonify({"text": text})

    except Exception as e:
        # If generation fails, return a safe fallback
        return jsonify({"text": f"Art encompasses diverse creative expressions that reflect human culture, emotion, and ideas across time and medium."})


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
