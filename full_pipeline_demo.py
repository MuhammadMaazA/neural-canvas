#!/usr/bin/env python3
"""
FULL PIPELINE: Image → CNN → LLM Explanation
=============================================
1. Stream real artwork image from WikiArt (NO DOWNLOAD)
2. CNN classifies it (artist, style, genre)
3. LLM explains the classification
"""
import os
# Set cache to project directory
os.environ['HF_HOME'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['HF_DATASETS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'

import sys
sys.path.append('/cs/student/projects1/2023/muhamaaz/neural-canvas')
sys.path.append('/cs/student/projects1/2023/muhamaaz/neural-canvas/llm')
sys.path.append('/cs/student/projects1/2023/muhamaaz/neural-canvas/cnn_models')

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from models.art_expert_model import create_art_expert_model
from cnn_models.model import build_model
from cnn_models.config import Config

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "="*70)
    print("🎨 FULL PIPELINE: Image → CNN → LLM")
    print("="*70)
    print("⚡ Using STREAMING (no large downloads)")
    print("="*70)
    
    # ========== STEP 1: Load CNN ==========
    print("\n[1/4] Loading CNN model...")
    config = Config()
    
    # Stream dataset just to get class names
    print("   Streaming WikiArt metadata...")
    dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
    artist_names = dataset.features['artist'].names
    style_names = dataset.features['style'].names
    genre_names = dataset.features['genre'].names
    print(f"   Found {len(artist_names)} artists, {len(style_names)} styles, {len(genre_names)} genres")
    
    num_classes = {
        'artist': len(artist_names),
        'style': len(style_names),
        'genre': len(genre_names)
    }
    
    cnn_model = build_model(config, num_classes).to(device)
    
    # Load CNN checkpoint
    cnn_checkpoint = "/cs/student/projects1/2023/muhamaaz/neural-canvas/cnn_models/checkpoints/best_multitask_macro0.6421.pt"
    if os.path.exists(cnn_checkpoint):
        ckpt = torch.load(cnn_checkpoint, map_location=device)
        cnn_model.load_state_dict(ckpt['model'])
        print(f"   ✅ CNN loaded (macro acc: {ckpt.get('macro_acc', 0):.2%})")
    else:
        print("   ⚠️ CNN checkpoint not found, using untrained")
    cnn_model.eval()
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ========== STEP 2: Load LLM models ==========
    print("\n[2/4] Loading LLM models...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    checkpoint = torch.load(
        "/cs/student/projects1/2023/muhamaaz/checkpoints/cnn_explainer_from_scratch/best_model.pt",
        map_location=device, weights_only=False
    )
    model1 = create_art_expert_model(tokenizer.vocab_size, "base").to(device)
    model1.load_state_dict(checkpoint['model_state_dict'])
    model1.eval()
    print("   ✅ Model 1 (From Scratch) ready")
    
    model2 = AutoModelForCausalLM.from_pretrained(
        "/cs/student/projects1/2023/muhamaaz/checkpoints/cnn_explainer_finetuned/best_model_hf"
    ).to(device)
    model2.eval()
    print("   ✅ Model 2 (Fine-tuned) ready")
    
    # ========== STEP 3: Stream images from WikiArt ==========
    print("\n[3/4] Streaming images from WikiArt (NO DOWNLOAD)...")
    test_dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
    
    # Get 3 samples by streaming (skip first 100 to get variety)
    samples = []
    for i, item in enumerate(test_dataset):
        if i < 100:  # Skip first 100
            continue
        if len(samples) >= 3:
            break
        samples.append(item)
    print(f"   ✅ Streamed {len(samples)} images")
    
    # ========== STEP 4: Process each image ==========
    print("\n[4/4] Processing images through pipeline...")
    
    for i, item in enumerate(samples, 1):
        print("\n" + "="*70)
        print(f"🖼️  IMAGE {i}/3")
        print("="*70)
        
        # Get ground truth
        gt_artist = artist_names[item['artist']]
        gt_style = style_names[item['style']]
        gt_genre = genre_names[item['genre']]
        
        print(f"\n📋 GROUND TRUTH:")
        print(f"   Artist: {gt_artist}")
        print(f"   Style:  {gt_style}")
        print(f"   Genre:  {gt_genre}")
        
        # CNN prediction
        image = item['image'].convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = cnn_model(img_tensor)
        
        # Get predictions
        print(f"\n🔍 CNN PREDICTION:")
        predictions = {}
        for task in ['artist', 'style', 'genre']:
            probs = F.softmax(logits[task], dim=1)
            conf, idx = torch.max(probs, dim=1)
            
            if task == 'artist':
                name = artist_names[idx.item()]
                gt = gt_artist
            elif task == 'style':
                name = style_names[idx.item()]
                gt = gt_style
            else:
                name = genre_names[idx.item()]
                gt = gt_genre
            
            predictions[task] = {'name': name, 'conf': conf.item()}
            match = '✓' if name.lower() == gt.lower() else '✗'
            print(f"   {task.capitalize()}: {name} ({conf.item():.1%}) {match}")
        
        # Create prompt for LLM
        artist = predictions['artist']['name']
        style = predictions['style']['name']
        genre = predictions['genre']['name']
        a_conf = predictions['artist']['conf']
        s_conf = predictions['style']['conf']
        g_conf = predictions['genre']['conf']
        
        prompt = f"""The CNN classified this artwork as:
- Artist: {artist} ({a_conf:.0%} confidence)
- Style: {style} ({s_conf:.0%} confidence)
- Genre: {genre} ({g_conf:.0%} confidence)

Explain this classification:"""

        tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
        
        # LLM explanations
        print(f"\n💬 LLM EXPLANATIONS:")
        
        with torch.no_grad():
            out1 = model1.generate(tokens, max_new_tokens=150, temperature=0.7)
            out2 = model2.generate(tokens, max_new_tokens=150, temperature=0.7, 
                                   do_sample=True, pad_token_id=tokenizer.eos_token_id)
        
        resp1 = tokenizer.decode(out1[0], skip_special_tokens=True)[len(prompt):].strip()
        resp2 = tokenizer.decode(out2[0], skip_special_tokens=True)[len(prompt):].strip()
        
        # Cut at last complete sentence
        def get_complete_sentences(text):
            # Find last period, question mark, or exclamation
            for end in ['. ', '! ', '? ']:
                last_idx = text.rfind(end)
                if last_idx > 50:
                    return text[:last_idx+1]
            return text
        
        print(f"\n🤖 MODEL 1 (From Scratch):")
        print("-"*50)
        print(get_complete_sentences(resp1))
        
        print(f"\n🎯 MODEL 2 (Fine-tuned):")
        print("-"*50)
        print(get_complete_sentences(resp2))
    
    print("\n" + "="*70)
    print("✅ FULL PIPELINE DEMO COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
