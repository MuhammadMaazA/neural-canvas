#!/usr/bin/env python3
"""
LLM Inference Test Script
=========================
Test your trained CNN explainer models interactively.
"""

import os
os.environ['HF_HOME'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append('/cs/student/projects1/2023/muhamaaz/neural-canvas/llm')
from models.art_expert_model import create_art_expert_model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")
    
    # Load tokenizer
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Model 1 (From Scratch)
    print("🔧 Loading Model 1 (From Scratch)...")
    checkpoint = torch.load(
        "/cs/student/projects1/2023/muhamaaz/checkpoints/cnn_explainer_from_scratch/best_model.pt",
        map_location=device, weights_only=False
    )
    model1 = create_art_expert_model(tokenizer.vocab_size, "base").to(device)
    model1.load_state_dict(checkpoint['model_state_dict'])
    model1.eval()
    print("   ✅ Model 1 ready (56M params)")
    
    # Load Model 2 (Fine-tuned)
    print("🔧 Loading Model 2 (Fine-tuned)...")
    model2 = AutoModelForCausalLM.from_pretrained(
        "/cs/student/projects1/2023/muhamaaz/checkpoints/cnn_explainer_finetuned/best_model_hf"
    ).to(device)
    model2.eval()
    print("   ✅ Model 2 ready (82M params)")
    
    print("\n" + "="*70)
    print("🎨 CNN EXPLAINER - INTERACTIVE TEST")
    print("="*70)
    print("Enter CNN predictions to see how each model explains them.")
    print("Type 'quit' to exit, 'demo' for sample predictions.")
    print("="*70)
    
    while True:
        print("\n" + "-"*70)
        cmd = input("Enter command (demo/quit) or artist name: ").strip()
        
        if cmd.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if cmd.lower() == 'demo':
            # Demo predictions
            samples = [
                ("Vincent van Gogh", "Post-Impressionism", "Landscape", 0.87, 0.92, 0.78),
                ("Pablo Picasso", "Cubism", "Portrait", 0.76, 0.88, 0.72),
                ("Claude Monet", "Impressionism", "Landscape", 0.91, 0.95, 0.85),
            ]
            for artist, style, genre, a, s, g in samples:
                run_inference(model1, model2, tokenizer, device, artist, style, genre, a, s, g)
            continue
        
        # Custom input
        artist = cmd if cmd else "Unknown Artist"
        style = input("Style: ").strip() or "Unknown"
        genre = input("Genre: ").strip() or "Unknown"
        
        try:
            a_conf = float(input("Artist confidence (0-1, default 0.8): ") or "0.8")
            s_conf = float(input("Style confidence (0-1, default 0.85): ") or "0.85")
            g_conf = float(input("Genre confidence (0-1, default 0.75): ") or "0.75")
        except:
            a_conf, s_conf, g_conf = 0.8, 0.85, 0.75
        
        run_inference(model1, model2, tokenizer, device, artist, style, genre, a_conf, s_conf, g_conf)


def run_inference(model1, model2, tokenizer, device, artist, style, genre, a_conf, s_conf, g_conf):
    """Run inference on both models"""
    
    prompt = f"""The CNN classified this artwork as:
- Artist: {artist} ({a_conf:.0%} confidence)
- Style: {style} ({s_conf:.0%} confidence)
- Genre: {genre} ({g_conf:.0%} confidence)

Explain this classification:"""

    print("\n" + "="*70)
    print("📊 CNN PREDICTION:")
    print(f"   Artist: {artist} ({a_conf:.0%})")
    print(f"   Style:  {style} ({s_conf:.0%})")
    print(f"   Genre:  {genre} ({g_conf:.0%})")
    print("="*70)
    
    tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    
    # Model 1
    print("\n🤖 MODEL 1 (From Scratch):")
    print("-"*50)
    with torch.no_grad():
        out1 = model1.generate(tokens, max_new_tokens=120, temperature=0.7)
    resp1 = tokenizer.decode(out1[0], skip_special_tokens=True)[len(prompt):].strip()
    print(resp1)
    
    # Model 2
    print("\n🎯 MODEL 2 (Fine-tuned):")
    print("-"*50)
    with torch.no_grad():
        out2 = model2.generate(tokens, max_new_tokens=120, temperature=0.7, 
                               do_sample=True, pad_token_id=tokenizer.eos_token_id)
    resp2 = tokenizer.decode(out2[0], skip_special_tokens=True)[len(prompt):].strip()
    print(resp2)
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

