#!/usr/bin/env python3
"""Quick LLM test with projects folder cache"""

import os
# Set cache to projects folder BEFORE imports
os.environ['HF_HOME'] = '/cs/student/projects1/2023/muhamaaz/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/.cache/huggingface'

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("="*80)
print("LLM MODELS TEST (Cache in Projects Folder)")
print("="*80)

# Sample CNN prediction
sample = {
    "artist": "Vincent van Gogh",
    "style": "Post-Impressionism",
    "genre": "Landscape"
}

print(f"\n📊 Test CNN Input:")
print(f"   Artist: {sample['artist']}")
print(f"   Style: {sample['style']}")
print(f"   Genre: {sample['genre']}")

def test_model(name, checkpoint_path):
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✅ Checkpoint loaded (Epoch {checkpoint.get('epoch', 'N/A')})")

        # Load model
        print("   Loading GPT-2 base model...")
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        print(f"✅ Model ready!")

        # Create prompt
        prompt = f"Artist: {sample['artist']}\nStyle: {sample['style']}\nGenre: {sample['genre']}\n\nExplanation:"

        print(f"\n   Generating text...")
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=150,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\n{'='*40}")
        print(f"Generated Text:")
        print(f"{'='*40}")
        print(text)
        print()

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Test both models
results = []

results.append(test_model(
    "Model 1 (From Scratch)",
    "/cs/student/projects1/2023/muhamaaz/checkpoints/cnn_explainer_from_scratch/best_model.pt"
))

results.append(test_model(
    "Model 2 (Fine-tuned GPT-2)",
    "/cs/student/projects1/2023/muhamaaz/checkpoints/cnn_explainer_finetuned/best_model.pt"
))

# Summary
print("="*80)
print("TEST SUMMARY")
print("="*80)
print(f"Model 1 (From Scratch): {'✅ WORKS' if results[0] else '❌ FAILED'}")
print(f"Model 2 (Fine-tuned):   {'✅ WORKS' if results[1] else '❌ FAILED'}")
print("="*80)
