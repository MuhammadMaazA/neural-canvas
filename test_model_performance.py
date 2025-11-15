"""
Comprehensive Evaluation of Model 1
Test with various prompts to see actual performance
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import sys
sys.path.insert(0, '/cs/student/projects1/2023/muhamaaz/code/neural-canvas')
from llm.models.modern_transformer import ModernTransformer
import time

def generate_response(model, tokenizer, prompt, max_length=100, temperature=0.7, device='cuda'):
    """Generate response from prompt"""
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(generated, None)
            next_token_logits = logits[0, -1, :] / temperature
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            if generated.shape[1] > 512:
                generated = generated[:, -512:]
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def main():
    print("="*80)
    print("MODEL 1 COMPREHENSIVE EVALUATION")
    print("="*80)
    
    # Load model
    print("\n[1/3] Loading model...")
    checkpoint_path = "/cs/student/projects1/2023/muhamaaz/checkpoints/model1/model1_final.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']

    # CRITICAL FIX: Use the SAME tokenizer as training!
    # The checkpoint stores which tokenizer was used during training
    tokenizer_name = checkpoint.get('tokenizer_name', 'meta-llama/Llama-2-7b-hf')
    print(f"   Using tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = ModernTransformer(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        n_kv_heads=config['n_kv_heads'],
        max_len=config['max_seq_len'],
        dropout=config['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded (Epoch {checkpoint['epoch']}, Loss {checkpoint['loss']:.4f})")
    print(f"✓ Device: {device}")
    
    # Test prompts
    print("\n[2/3] Testing with various prompts...")
    print("="*80)
    
    test_prompts = [
        # Simple questions
        ("What is AI?", "Simple Q&A"),
        ("How does machine learning work?", "Technical Q&A"),
        ("Explain neural networks.", "Explanation request"),
        
        # Art-related (trained domain)
        ("Describe impressionist art.", "Art domain"),
        ("Who was Pablo Picasso?", "Art knowledge"),
        ("What is abstract expressionism?", "Art movement"),
        
        # Conversational
        ("Hello, how are you?", "Greeting"),
        ("Can you help me understand deep learning?", "Help request"),
        ("Tell me about Vincent van Gogh.", "Biographical"),
        
        # Code-related (from StackOverflow)
        ("How do I use Python?", "Programming"),
        ("Explain loops in programming.", "Code concept"),
        
        # Complex
        ("What is the relationship between artificial intelligence and art?", "Complex reasoning"),
        ("How do convolutional neural networks process images?", "Technical detail"),
    ]
    
    results = []
    for i, (prompt, category) in enumerate(test_prompts):
        print(f"\n--- Test {i+1}/{len(test_prompts)}: {category} ---")
        print(f"Prompt: \"{prompt}\"")
        
        start_time = time.time()
        response = generate_response(model, tokenizer, prompt, max_length=80, temperature=0.7, device=device)
        elapsed = time.time() - start_time
        
        # Extract only the generated part
        if response.startswith(prompt):
            generated = response[len(prompt):].strip()
        else:
            generated = response
        
        print(f"\nGenerated ({elapsed:.2f}s):")
        print(f"{generated[:200]}")  # First 200 chars
        
        # Score quality (very basic)
        coherence_score = 0
        if len(generated) > 10:
            coherence_score += 1
        if not any(weird in generated.lower() for weird in ['ġ', 'ċ', '�', 'endoftext']):
            coherence_score += 1
        if len(generated.split()) > 5:
            coherence_score += 1
        
        results.append({
            'prompt': prompt,
            'category': category,
            'response': generated,
            'coherence': coherence_score,
            'time': elapsed
        })
        
        print(f"Coherence score: {coherence_score}/3")
        print("-" * 80)
    
    # Summary
    print("\n[3/3] EVALUATION SUMMARY")
    print("="*80)
    
    avg_coherence = sum(r['coherence'] for r in results) / len(results)
    avg_time = sum(r['time'] for r in results) / len(results)
    
    print(f"\nOverall Statistics:")
    print(f"  Average coherence: {avg_coherence:.2f}/3.0")
    print(f"  Average generation time: {avg_time:.2f}s")
    
    # Category breakdown
    print(f"\nBy Category:")
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r['coherence'])
    
    for cat, scores in categories.items():
        avg = sum(scores) / len(scores)
        print(f"  {cat:20s}: {avg:.2f}/3.0")
    
    # Problems found
    print(f"\nCommon Issues:")
    issues = {
        'Too short': sum(1 for r in results if len(r['response']) < 20),
        'Incoherent': sum(1 for r in results if r['coherence'] < 2),
        'Off-topic': sum(1 for r in results if len(r['response'].split()) < 5),
    }
    
    for issue, count in issues.items():
        if count > 0:
            print(f"  ❌ {issue}: {count}/{len(results)} responses")
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    
    if avg_coherence < 1.5:
        print("❌ Model performs POORLY - needs significant fine-tuning or retraining")
        print("   Suggested actions:")
        print("   1. Fine-tune with lower LR (1e-5) for more epochs (10-20)")
        print("   2. Add more diverse training data")
        print("   3. Increase training epochs (currently only 9)")
    elif avg_coherence < 2.5:
        print("⚠️  Model performs OKAY - fine-tuning recommended")
        print("   Suggested actions:")
        print("   1. Fine-tune with LR 3e-5 for 5-10 epochs")
        print("   2. Focus on specific domains (art, technical, etc.)")
    else:
        print("✅ Model performs WELL - minor improvements possible")
        print("   Optional actions:")
        print("   1. Light fine-tuning for specific use cases")
        print("   2. Deploy as-is for testing")
    
    print("="*80)

if __name__ == "__main__":
    main()
