#!/usr/bin/env python3
"""
Test the new WikiArt-trained model
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from models.art_expert_model import ArtExpertTransformer

# Paths
CHECKPOINT_PATH = "/cs/student/projects1/2023/muhamaaz/checkpoints/model1_custom_v2_fixed_wikiart/best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Load the trained model"""
    print("Loading model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    
    # Get model config from checkpoint
    config = checkpoint.get('config', {})
    vocab_size = config.get('vocab_size', 50257)
    
    # Create model
    model = ArtExpertTransformer(
        vocab_size=vocab_size,
        dim=config.get('dim', 512),
        n_layers=config.get('n_layers', 8),
        n_heads=config.get('n_heads', 8),
        n_kv_heads=config.get('n_kv_heads', 2),
        max_len=config.get('max_len', 512),
        dropout=config.get('dropout', 0.15)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    val_loss = checkpoint.get('best_val_loss', None)
    
    if val_loss is not None:
        print(f"✓ Model ready! (Epoch {epoch}, Val Loss: {val_loss:.4f})")
    else:
        print(f"✓ Model ready! (Epoch {epoch})")
    print(f"This model was trained on REAL art data (WikiArt)!")
    
    return model, tokenizer


def generate_response(model, tokenizer, question, max_length=150, temperature=0.8):
    """Generate a response to a question"""
    prompt = f"Q: {question}\n\nA:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer part
    if "\n\nA:" in response:
        answer = response.split("\n\nA:")[-1].strip()
    else:
        answer = response.replace(prompt, "").strip()
    
    return answer


def main():
    """Interactive chat"""
    model, tokenizer = load_model()
    
    print("\n" + "="*60)
    print("ART EXPERT CHATBOT (WikiArt-Trained Model)")
    print("="*60)
    print("Ask me about art, artists, styles, or techniques!")
    print("Type 'quit' or 'q' to exit")
    print("="*60 + "\n")
    
    test_questions = [
        "Who is Vincent van Gogh?",
        "What is Impressionism?",
        "Tell me about Leonardo da Vinci",
        "What is the Baroque style?",
        "Describe Renaissance art"
    ]
    
    while True:
        print("\nSuggested questions:")
        for i, q in enumerate(test_questions[:3], 1):
            print(f"  {i}. {q}")
        
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'q', 'exit']:
            print("\nGoodbye! 🎨")
            break
        
        if not question:
            continue
        
        print(f"\nAnswer: ", end="", flush=True)
        answer = generate_response(model, tokenizer, question)
        print(answer)
        print("-" * 60)


if __name__ == "__main__":
    main()
