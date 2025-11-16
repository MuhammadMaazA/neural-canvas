#!/usr/bin/env python3
"""
Simple Chat - Ask your own questions!
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from models.art_expert_model import create_art_expert_model

# Load model
print("Loading model...", flush=True)
checkpoint_path = '/cs/student/projects1/2023/muhamaaz/checkpoints/model1_custom/best_model.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = len(tokenizer)
model = create_art_expert_model(vocab_size=vocab_size, model_size="base")
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"✓ Model ready! (Epoch {checkpoint.get('epoch', '?')})")
print()
print("=" * 60)
print("ASK ME ANYTHING ABOUT ART!")
print("=" * 60)
print("Type your question and press Enter")
print("Type 'quit' or 'q' to exit")
print("=" * 60)
print()

while True:
    # Get question
    question = input("Your question: ").strip()
    
    # Check if user wants to quit
    if question.lower() in ['quit', 'q', 'exit', 'bye']:
        print("\nGoodbye! 👋\n")
        break
    
    if not question:
        continue
    
    # Format as Q&A
    prompt = f"Q: {question}\nA:"
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
    input_ids = inputs['input_ids'].to(device)
    
    print("\nAnswer: ", end="", flush=True)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer
    if "A:" in response:
        answer = response.split("A:", 1)[1].strip()
        if "\nQ:" in answer:
            answer = answer.split("\nQ:")[0].strip()
        print(answer)
    else:
        print(response)
    
    print("\n" + "-" * 60 + "\n")
