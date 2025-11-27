#!/usr/bin/env python3
"""
Quick inference test for latest checkpoint
"""
import torch
import sys
sys.path.append('/cs/student/projects1/2023/muhamaaz/neural-canvas')

from transformers import GPT2Tokenizer
from llm.models.art_expert_model import ArtExpertTransformer

# Model config (must match training)
MODEL_CONFIG = {
    'vocab_size': 50257,
    'dim': 512,
    'n_layers': 8,
    'n_heads': 8,
    'n_kv_heads': 2,
    'max_len': 512,
    'dropout': 0.1
}

CHECKPOINT_PATH = '/cs/student/projects1/2023/muhamaaz/checkpoints/model1_custom_v2_fixed_wikiart/checkpoint_epoch_16.pt'

def test_inference():
    print("🎨 Testing Neural Canvas - Epoch 4 Checkpoint\n")
    print("=" * 70)
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ArtExpertTransformer(**MODEL_CONFIG).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded checkpoint from Epoch {checkpoint.get('epoch', 'unknown')}")
    val_loss = checkpoint.get('val_loss', None)
    if val_loss is not None:
        print(f"✓ Validation Loss: {val_loss:.4f}")
    else:
        print(f"✓ Validation Loss: N/A")
    print(f"✓ Device: {device}")
    print()
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test questions
    test_questions = [
        "Art",
        "Who was Vincent van Gogh?",
        "What is Impressionism?",
        "Tell me about the Mona Lisa.",
        "What techniques did Rembrandt use?",
        "Describe Post-Impressionism.",
    ]
    
    print("🧪 TESTING INFERENCE")
    print("=" * 70)
    print()
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 70)
        
        # Tokenize
        inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True, max_length=256)
        input_ids = inputs['input_ids'].to(device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=150,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
        
        # Decode
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Extract only the generated part (after the question)
        if question in response:
            answer = response[len(question):].strip()
        else:
            answer = response.strip()
        
        print(f"Answer: {answer}")
        print()
    
    print("=" * 70)
    print("✓ Inference test complete!")
    print("\n💡 Note: Model has trained for 4 epochs with FIXED WikiArt dataset (81K art samples)")
    print("   Quality should be improving - compare with earlier epochs!")

if __name__ == "__main__":
    test_inference()
