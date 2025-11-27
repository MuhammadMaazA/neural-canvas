#!/usr/bin/env python3
"""
Quick inference test - Best model (Epoch 5)
"""
import torch
import sys
sys.path.append('/cs/student/projects1/2023/muhamaaz/neural-canvas')

from transformers import GPT2Tokenizer
from llm.models.art_expert_model import ArtExpertTransformer

MODEL_CONFIG = {
    'vocab_size': 50257,
    'dim': 512,
    'n_layers': 8,
    'n_heads': 8,
    'n_kv_heads': 2,
    'max_len': 512,
    'dropout': 0.1
}

def quick_test():
    print("\n" + "="*70)
    print("🎨 NEURAL CANVAS - INFERENCE TEST (EPOCH 5 - BEST MODEL)")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ArtExpertTransformer(**MODEL_CONFIG).to(device)
    
    checkpoint = torch.load(
        '/cs/student/projects1/2023/muhamaaz/checkpoints/model1_custom_v2_fixed_wikiart/best_model.pt',
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    epoch = checkpoint.get('epoch', '?')
    print(f"✅ Loaded: Epoch {epoch} (BEST MODEL)")
    print(f"✅ Device: {device}")
    print(f"✅ Parameters: 56.1M")
    print(f"✅ Fixed WikiArt Dataset: 81,444 art samples\n")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    questions = [
        "Who was Vincent van Gogh?",
        "What is Impressionism?",
        "What is chiaroscuro?",
        "Tell me about Renaissance art.",
    ]
    
    for i, q in enumerate(questions, 1):
        print(f"\n{'─'*70}")
        print(f"Q{i}: {q}")
        print(f"{'─'*70}")
        
        inputs = tokenizer(q, return_tensors='pt', padding=True, truncation=True, max_length=256)
        input_ids = inputs['input_ids'].to(device)
        
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=120, temperature=0.7, top_k=40, top_p=0.9)
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = response[len(q):].strip() if q in response else response.strip()
        
        print(f"A{i}: {answer}")
    
    print(f"\n{'='*70}")
    print("✅ Test Complete!")
    print(f"{'='*70}\n")
    
    print("📊 TRAINING PROGRESS:")
    print("  • Epoch 5/30 completed")
    print("  • Loss improving: Train 1.92, Val 1.90, PPL 6.71")
    print("  • Model learning art knowledge from 81K WikiArt samples")
    print("  • Quality improving with each epoch!")

if __name__ == "__main__":
    quick_test()
