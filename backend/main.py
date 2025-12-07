#!/usr/bin/env python3
"""
Neural Canvas Backend API (FastAPI - DEPRECATED)
================================================
NOTE: This FastAPI backend is DEPRECATED. 
The primary backend is now Flask at: frontend/backend/app.py

This file is kept for reference only.
"""

import os
os.environ['HF_HOME'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import io
import sys
import numpy as np

# Add paths
sys.path.append('/cs/student/projects1/2023/muhamaaz/neural-canvas')
sys.path.append('/cs/student/projects1/2023/muhamaaz/neural-canvas/llm')
sys.path.append('/cs/student/projects1/2023/muhamaaz/neural-canvas/cnn_models')

from transformers import AutoTokenizer, AutoModelForCausalLM
from models.art_expert_model import create_art_expert_model
from cnn_models.model import build_model
from cnn_models.config import Config

app = FastAPI(title="Neural Canvas API", version="1.0.0")

# CORS - Allow frontend to access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models (loaded on startup)
cnn_model_scratch = None  # CNN from scratch
cnn_model_finetuned = None  # Fine-tuned ResNet50
llm_model1 = None
llm_model2 = None
tokenizer = None
device = None
artist_names = None
style_names = None
genre_names = None


class CNNPrediction(BaseModel):
    artist: str
    artist_confidence: float
    style: str
    style_confidence: float
    genre: str
    genre_confidence: float


class LLMExplanation(BaseModel):
    model: str  # "model1" or "model2"
    explanation: str


class FullResponse(BaseModel):
    predictions: CNNPrediction
    explanations: List[LLMExplanation]


class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = "both"  # "model1", "model2", or "both"
    max_tokens: int = 150


def load_models():
    """Load all models on startup"""
    global cnn_model_scratch, cnn_model_finetuned, llm_model1, llm_model2, tokenizer, device, artist_names, style_names, genre_names
    
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
    
    # Load CNN from scratch
    print("Loading CNN from scratch...")
    config = Config()
    cnn_model_scratch = build_model(config, num_classes).to(device)
    cnn_checkpoint_scratch = "/cs/student/projects1/2023/muhamaaz/neural-canvas/cnn_models/checkpoints/best_multitask_macro0.6421.pt"
    if os.path.exists(cnn_checkpoint_scratch):
        ckpt = torch.load(cnn_checkpoint_scratch, map_location=device)
        cnn_model_scratch.load_state_dict(ckpt['model'])
        print(f"✓ CNN from scratch loaded (macro acc: {ckpt.get('macro_acc', 0):.2%})")
    cnn_model_scratch.eval()
    
    # Load fine-tuned ResNet50
    print("Loading fine-tuned ResNet50...")
    try:
        # Import the actual model class from training script
        import sys
        sys.path.insert(0, '/cs/student/projects1/2023/muhamaaz/neural-canvas/cnn_models')
        from train_finetuned_cnn import PretrainedMultiTaskCNN
        
        # Create model with same architecture as training
        cnn_model_finetuned = PretrainedMultiTaskCNN(
            num_artists=num_classes['artist'],
            num_styles=num_classes['style'],
            num_genres=num_classes['genre'],
            backbone="resnet50",
            pretrained=False  # We'll load from checkpoint, but structure should match
        ).to(device)
        
        # Find best fine-tuned checkpoint
        import glob
        finetuned_checkpoints = glob.glob("/cs/student/projects1/2023/muhamaaz/neural-canvas/cnn_models/checkpoints_finetuned/best_finetuned_macro*.pt")
        if finetuned_checkpoints:
            best_ckpt = max(finetuned_checkpoints, key=lambda x: float(x.split('macro')[1].split('.pt')[0]))
            print(f"Loading checkpoint: {best_ckpt}")
            ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
            
            # Handle different checkpoint formats
            if 'model' in ckpt:
                cnn_model_finetuned.load_state_dict(ckpt['model'])
            elif 'model_state_dict' in ckpt:
                cnn_model_finetuned.load_state_dict(ckpt['model_state_dict'])
            else:
                cnn_model_finetuned.load_state_dict(ckpt)
            
            macro_acc = ckpt.get('macro_acc', ckpt.get('best_macro_acc', 0))
            print(f"✓ Fine-tuned ResNet50 loaded (macro acc: {macro_acc:.2%})")
            cnn_model_finetuned.eval()
        else:
            print("⚠ Fine-tuned checkpoint not found")
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
    
    # Load LLM Model 2 (Fine-tuned) - try GPT-2 Medium first, fallback to DistilGPT-2
    print("Loading LLM Model 2 (Fine-tuned)...")
    model2_path = "/cs/student/projects1/2023/muhamaaz/checkpoints/model2_cnn_explainer_gpt2medium/best_model_hf"
    if not os.path.exists(model2_path):
        model2_path = "/cs/student/projects1/2023/muhamaaz/checkpoints/cnn_explainer_finetuned/best_model_hf"
    
    if os.path.exists(model2_path):
        llm_model2 = AutoModelForCausalLM.from_pretrained(model2_path).to(device)
        llm_model2.eval()
        print("✓ Model 2 loaded")
    else:
        print("⚠ Model 2 not found, using Model 1 only")
        llm_model2 = None
    
    print("✅ All models loaded!")


@app.on_event("startup")
async def startup_event():
    load_models()


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for CNN"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image.convert('RGB')).unsqueeze(0)


def predict_cnn(image: Image.Image, model_type: str = "scratch") -> CNNPrediction:
    """Run CNN prediction on image
    
    Args:
        image: PIL Image
        model_type: "scratch" or "finetuned"
    """
    global cnn_model_scratch, cnn_model_finetuned, device, artist_names, style_names, genre_names
    
    model = cnn_model_scratch if model_type == "scratch" else cnn_model_finetuned
    if model is None:
        raise HTTPException(status_code=503, detail=f"CNN model {model_type} not loaded")
    
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
    
    return CNNPrediction(
        artist=predictions['artist']['name'],
        artist_confidence=predictions['artist']['confidence'],
        style=predictions['style']['name'],
        style_confidence=predictions['style']['confidence'],
        genre=predictions['genre']['name'],
        genre_confidence=predictions['genre']['confidence']
    )


def generate_explanation(prediction: CNNPrediction, model_num: int = 1, max_tokens: int = 150) -> str:
    """Generate LLM explanation for CNN prediction"""
    global llm_model1, llm_model2, tokenizer, device
    
    model = llm_model1 if model_num == 1 else llm_model2
    if model is None:
        return "Model not available"
    
    prompt = f"""The CNN classified this artwork as:
- Artist: {prediction.artist} ({prediction.artist_confidence:.0%} confidence)
- Style: {prediction.style} ({prediction.style_confidence:.0%} confidence)
- Genre: {prediction.genre} ({prediction.genre_confidence:.0%} confidence)

Explain this classification:"""

    tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    
    with torch.no_grad():
        if model_num == 1:
            output = model.generate(tokens, max_new_tokens=max_tokens, temperature=0.7)
        else:
            output = model.generate(
                tokens, max_new_tokens=max_tokens, temperature=0.7,
                do_sample=True, pad_token_id=tokenizer.eos_token_id
            )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    explanation = response[len(prompt):].strip()
    
    # Get complete sentences
    for end in ['. ', '! ', '? ']:
        last_idx = explanation.rfind(end)
        if last_idx > 50:
            return explanation[:last_idx+1]
    
    return explanation


@app.get("/")
async def root():
    return {
        "message": "Neural Canvas API",
        "version": "1.0.0",
        "endpoints": {
            "/classify": "POST - Classify image with CNN",
            "/explain": "POST - Get LLM explanation",
            "/full": "POST - Full pipeline (CNN + LLM)",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "cnn_scratch": cnn_model_scratch is not None,
            "cnn_finetuned": cnn_model_finetuned is not None,
            "llm_model1": llm_model1 is not None,
            "llm_model2": llm_model2 is not None
        },
        "device": str(device)
    }


@app.post("/classify", response_model=CNNPrediction)
async def classify_image(
    file: UploadFile = File(...),
    model_type: str = "scratch"  # "scratch" or "finetuned"
):
    """
    Classify artwork image using CNN
    
    Args:
        file: Image file
        model_type: "scratch" (from scratch) or "finetuned" (ResNet50)
    
    Returns: Artist, Style, Genre with confidence scores
    """
    if model_type == "scratch" and cnn_model_scratch is None:
        raise HTTPException(status_code=503, detail="CNN from scratch not loaded")
    if model_type == "finetuned" and cnn_model_finetuned is None:
        raise HTTPException(status_code=503, detail="Fine-tuned CNN not loaded")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Predict
        prediction = predict_cnn(image, model_type)
        
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/generate", response_model=List[LLMExplanation])
async def generate_text(request: GenerateRequest):
    """
    Generate text from LLM models given a prompt
    
    Args:
        prompt: Text prompt to generate from
        model: Which model(s) to use ("model1", "model2", or "both")
        max_tokens: Maximum tokens to generate
    """
    if llm_model1 is None:
        raise HTTPException(status_code=503, detail="LLM models not loaded")
    
    explanations = []
    
    if request.model in ["model1", "both"]:
        tokens = tokenizer(request.prompt, return_tensors='pt')['input_ids'].to(device)
        with torch.no_grad():
            output = llm_model1.generate(tokens, max_new_tokens=request.max_tokens, temperature=0.7)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        explanation = response[len(request.prompt):].strip()
        explanations.append(LLMExplanation(model="model1", explanation=explanation))
    
    if request.model in ["model2", "both"] and llm_model2 is not None:
        tokens = tokenizer(request.prompt, return_tensors='pt')['input_ids'].to(device)
        with torch.no_grad():
            output = llm_model2.generate(
                tokens, max_new_tokens=request.max_tokens, temperature=0.7,
                do_sample=True, pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        explanation = response[len(request.prompt):].strip()
        explanations.append(LLMExplanation(model="model2", explanation=explanation))
    
    return explanations


@app.post("/explain", response_model=List[LLMExplanation])
async def explain_classification(
    artist: str,
    style: str,
    genre: str,
    artist_confidence: float,
    style_confidence: float,
    genre_confidence: float,
    model: Optional[str] = "both"  # "model1", "model2", or "both"
):
    """
    Generate LLM explanation for CNN predictions
    
    Args:
        artist, style, genre: Classification results
        artist_confidence, etc: Confidence scores (0-1)
        model: Which model(s) to use ("model1", "model2", or "both")
    """
    if llm_model1 is None:
        raise HTTPException(status_code=503, detail="LLM models not loaded")
    
    prediction = CNNPrediction(
        artist=artist,
        artist_confidence=artist_confidence,
        style=style,
        style_confidence=style_confidence,
        genre=genre,
        genre_confidence=genre_confidence
    )
    
    explanations = []
    
    if model in ["model1", "both"]:
        exp1 = generate_explanation(prediction, model_num=1)
        explanations.append(LLMExplanation(model="model1", explanation=exp1))
    
    if model in ["model2", "both"] and llm_model2 is not None:
        exp2 = generate_explanation(prediction, model_num=2)
        explanations.append(LLMExplanation(model="model2", explanation=exp2))
    
    return explanations


@app.post("/classify-both")
async def classify_both_cnns(file: UploadFile = File(...)):
    """
    Classify with BOTH CNN models for comparison
    
    Returns: Predictions from both models
    """
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        scratch_pred = predict_cnn(image, "scratch")
        finetuned_pred = predict_cnn(image, "finetuned") if cnn_model_finetuned else None
        
        return {
            "scratch": scratch_pred,
            "finetuned": finetuned_pred
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/full", response_model=FullResponse)
async def full_pipeline(
    file: UploadFile = File(...),
    cnn_model: str = "scratch"  # Which CNN to use
):
    """
    Full pipeline: Image → CNN → LLM Explanation
    
    Args:
        file: Image file
        cnn_model: "scratch" or "finetuned"
    
    Returns: CNN predictions + LLM explanations from both models
    """
    if (cnn_model == "scratch" and cnn_model_scratch is None) or \
       (cnn_model == "finetuned" and cnn_model_finetuned is None):
        raise HTTPException(status_code=503, detail="CNN model not loaded")
    if llm_model1 is None:
        raise HTTPException(status_code=503, detail="LLM models not loaded")
    
    try:
        # CNN Classification
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        prediction = predict_cnn(image, cnn_model)
        
        # LLM Explanations
        explanations = []
        
        exp1 = generate_explanation(prediction, model_num=1)
        explanations.append(LLMExplanation(model="model1", explanation=exp1))
        
        if llm_model2 is not None:
            exp2 = generate_explanation(prediction, model_num=2)
            explanations.append(LLMExplanation(model="model2", explanation=exp2))
        
        return FullResponse(
            predictions=prediction,
            explanations=explanations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in pipeline: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

