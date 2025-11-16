#!/usr/bin/env python3
"""
Quick test of model responses - non-interactive
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from models.art_expert_model import create_art_expert_model

print("=" * 80)
print("TESTING MODEL RESPONSES")
print("=" * 80)

# Load checkpoint
checkpoint_path = '/cs/student/projects1/2023/muhamaaz/checkpoints/model1_custom/best_model.pt'
print(f"\nLoading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Create model
print("Loading model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = len(tokenizer)
model = create_art_expert_model(vocab_size=vocab_size, model_size="base")
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"✓ Model loaded on {device}\n")

# Test prompts
test_prompts = [
    "Q: What is Impressionism?\nA:",
    "Q: Tell me about Renaissance art.\nA:",
    "Q: What is abstract art?\nA:",
    "Q: Who was Vincent van Gogh?\nA:",
    "Q: Explain what makes art beautiful.\nA:",
]

print("=" * 80)
print("GENERATING RESPONSES")
print("=" * 80)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n[Test {i}]")
    print(f"Prompt: {prompt.split('A:')[0].strip()}")
    
    # Encode
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
    input_ids = inputs['input_ids'].to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=80,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer
    if "A:" in response:
        answer = response.split("A:", 1)[1].strip()
        if "\nQ:" in answer:
            answer = answer.split("\nQ:")[0].strip()
        print(f"Answer: {answer}")
    else:
        print(f"Response: {response}")
    
    print("-" * 80)

print("\n" + "=" * 80)
print("✅ TESTING COMPLETE")
print("=" * 80)
