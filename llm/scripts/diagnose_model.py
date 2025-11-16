"""
Quick Diagnostic Script to Find Why Model Outputs Are Garbage
"""

import os
os.environ['HF_HOME'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['HF_DATASETS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'

import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from models.art_expert_model import create_art_expert_model

print("=" * 80)
print("MODEL DIAGNOSTIC - WHY ARE OUTPUTS GARBAGE?")
print("=" * 80)

# Load model and tokenizer
checkpoint_path = "/cs/student/projects1/2023/muhamaaz/checkpoints/model1_custom/best_model.pt"
print(f"\nLoading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
print(f"✓ Checkpoint loaded")
print(f"  Epoch: {checkpoint['epoch'] + 1}")
print(f"  Val Loss: {checkpoint.get('loss', 'N/A')}")

# Load tokenizer
tokenizer_name = checkpoint['config'].get('tokenizer_name', 'gpt2')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token
print(f"\n✓ Tokenizer loaded: {tokenizer_name}")
print(f"  Vocab size: {tokenizer.vocab_size}")

# Create and load model
model_size = checkpoint['config'].get('model_size', 'base')
model = create_art_expert_model(tokenizer.vocab_size, model_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"\n✓ Model loaded")

# TEST 1: Check if model can predict simple patterns
print("\n" + "=" * 80)
print("TEST 1: Can model predict next token for simple text?")
print("=" * 80)

test_texts = [
    "Q: What is Impressionism?\nA:",
    "The art style known as",
    "Vincent van Gogh was a"
]

with torch.no_grad():
    for text in test_texts:
        print(f"\nInput: '{text}'")

        # Tokenize
        inputs = tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids']

        print(f"  Token IDs: {input_ids[0].tolist()}")

        # Get logits
        logits, _ = model(input_ids)

        # Get top 5 predictions for next token
        next_token_logits = logits[0, -1, :]
        top_k = 5
        top_tokens = torch.topk(next_token_logits, top_k)

        print(f"  Top {top_k} predicted next tokens:")
        for i, (score, token_id) in enumerate(zip(top_tokens.values, top_tokens.indices)):
            token_text = tokenizer.decode([token_id.item()])
            print(f"    {i+1}. '{token_text}' (score: {score.item():.2f})")

# TEST 2: Check the training loss breakdown
print("\n" + "=" * 80)
print("TEST 2: Training history analysis")
print("=" * 80)

train_losses = checkpoint.get('train_losses', [])
val_losses = checkpoint.get('val_losses', [])
val_perps = checkpoint.get('val_perplexities', [])

if train_losses and val_losses:
    print(f"\nEpochs trained: {len(train_losses)}")
    print(f"\nLast 5 epochs:")
    print(f"  {'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Perplexity':<12} {'Overfitting?'}")
    print(f"  {'-'*70}")

    for i in range(max(0, len(train_losses) - 5), len(train_losses)):
        epoch = i + 1
        train_loss = train_losses[i]
        val_loss = val_losses[i]
        perp = val_perps[i] if i < len(val_perps) else 0
        gap = val_loss - train_loss
        overfitting = "YES" if gap > 0.5 else "No"

        print(f"  {epoch:<8} {train_loss:<12.4f} {val_loss:<12.4f} {perp:<12.2f} {overfitting}")

    print(f"\nFinal metrics:")
    print(f"  Best val loss: {min(val_losses):.4f}")
    print(f"  Best perplexity: {min(val_perps):.2f}")
    print(f"  Training-validation gap: {val_losses[-1] - train_losses[-1]:.4f}")

# TEST 3: Check actual generation
print("\n" + "=" * 80)
print("TEST 3: Generation test (what you're seeing)")
print("=" * 80)

prompts = [
    "Q: What is Impressionism?\nA:",
    "Q: Who was Vincent van Gogh?\nA:",
]

for prompt in prompts:
    print(f"\nPrompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']

    # Generate with different settings
    print("\n  [Low temperature (0.3) - should be focused]")
    output_ids = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 30,
        temperature=0.3,
        top_k=10,
        top_p=0.9,
        num_return_sequences=1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = response[len(prompt):].strip()
    print(f"  Answer: {answer[:100]}")

    print("\n  [Greedy (temperature=1.0, top_k=1) - most likely tokens]")
    output_ids = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 30,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        num_return_sequences=1,
        do_sample=False,  # Greedy
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = response[len(prompt):].strip()
    print(f"  Answer: {answer[:100]}")

# TEST 4: Check if there's a format mismatch
print("\n" + "=" * 80)
print("TEST 4: Training data format check")
print("=" * 80)

print("\nThe model was trained on texts like:")
print("  'Q: What is X?\\nA: X is...'")
print("\nBut during generation, are we using the same format?")
print("\nLet's check token alignment:")

training_format = "Q: What is Impressionism?\nA: Impressionism is an art movement"
inference_format = "Q: What is Impressionism?\nA:"

print(f"\nTraining format tokens:")
train_tokens = tokenizer(training_format, return_tensors='pt')
print(f"  {tokenizer.convert_ids_to_tokens(train_tokens['input_ids'][0].tolist())[:20]}")

print(f"\nInference format tokens:")
infer_tokens = tokenizer(inference_format, return_tensors='pt')
print(f"  {tokenizer.convert_ids_to_tokens(infer_tokens['input_ids'][0].tolist())}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nKey things to check:")
print("1. Are the top predicted tokens making ANY sense?")
print("2. Is the training-validation gap huge? (overfitting)")
print("3. Is perplexity extremely high? (not learning)")
print("4. Are tokens being predicted randomly?")
print("=" * 80)
