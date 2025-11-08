"""
Model Evaluation Utilities
UCL COMP0220 Coursework - Perplexity, BLEU, and Hallucination Tracking
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import re


def calculate_perplexity(model, dataloader, device, vocab_size: int) -> float:
    """
    Calculate perplexity on a dataset
    Perplexity = exp(cross_entropy_loss)
    
    Args:
        model: Trained language model
        dataloader: DataLoader for evaluation
        device: torch device
        vocab_size: Vocabulary size
    Returns:
        Perplexity score (lower is better)
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits, loss = model(inputs, targets)
            
            if loss is not None:
                # Count non-padding tokens
                mask = (targets != 0).float()
                num_tokens = mask.sum().item()
                
                if num_tokens > 0:
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


def calculate_bleu_score(predicted: List[str], reference: List[str], n: int = 4) -> float:
    """
    Calculate BLEU score for generated text
    
    Args:
        predicted: List of predicted sentences
        reference: List of reference sentences
        n: Maximum n-gram order
    Returns:
        BLEU score (0-1, higher is better)
    """
    def get_ngrams(text: str, n: int) -> Counter:
        words = text.lower().split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(tuple(words[i:i+n]))
        return Counter(ngrams)
    
    if len(predicted) != len(reference):
        return 0.0
    
    precisions = []
    
    for order in range(1, n + 1):
        matches = 0
        total_pred = 0
        
        for pred, ref in zip(predicted, reference):
            pred_ngrams = get_ngrams(pred, order)
            ref_ngrams = get_ngrams(ref, order)
            
            for ngram in pred_ngrams:
                total_pred += pred_ngrams[ngram]
                matches += min(pred_ngrams[ngram], ref_ngrams.get(ngram, 0))
        
        if total_pred == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / total_pred)
    
    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0
    
    bleu = np.exp(np.mean([np.log(p) for p in precisions]))
    
    # Brevity penalty
    pred_length = sum(len(p.split()) for p in predicted)
    ref_length = sum(len(r.split()) for r in reference)
    
    if pred_length < ref_length:
        bp = np.exp(1 - ref_length / pred_length) if pred_length > 0 else 0
    else:
        bp = 1.0
    
    return bleu * bp


def generate_text(model, vocab: Dict[str, int], prompt: str, max_length: int = 50, 
                  temperature: float = 1.0, device='cuda') -> str:
    """
    Generate text from a prompt
    
    Args:
        model: Trained language model
        vocab: Vocabulary dictionary
        prompt: Starting prompt text
        max_length: Maximum generation length
        temperature: Sampling temperature (higher = more random)
        device: torch device
    Returns:
        Generated text string
    """
    model.eval()
    
    # Reverse vocabulary for decoding
    idx_to_word = {idx: word for word, idx in vocab.items()}
    
    # Tokenize prompt
    words = re.findall(r'\b\w+\b', prompt.lower())
    tokens = [vocab.get(word, 3) for word in words]  # 3 = <UNK>
    tokens = torch.tensor([tokens], dtype=torch.long).to(device)
    
    generated_tokens = tokens.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions
            logits, _ = model(tokens, None)
            
            # Get logits for last token
            next_token_logits = logits[0, -1, :] / temperature
            
            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Stop if EOS token
            if next_token == 2:  # <EOS>
                break
            
            # Append to sequence
            generated_tokens = torch.cat([generated_tokens, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
            tokens = generated_tokens
    
    # Decode tokens to text
    generated_words = [idx_to_word.get(idx.item(), '<UNK>') for idx in generated_tokens[0]]
    generated_text = ' '.join([w for w in generated_words if w not in ['<PAD>', '<SOS>']])
    
    return generated_text


def detect_hallucinations(generated: str, source_texts: List[str], threshold: float = 0.3) -> Dict:
    """
    Detect potential hallucinations in generated text
    Simple approach: check if key phrases appear in source texts
    
    Args:
        generated: Generated text to check
        source_texts: List of source training texts
        threshold: Similarity threshold for detection
    Returns:
        Dictionary with hallucination metrics
    """
    generated_words = set(re.findall(r'\b\w+\b', generated.lower()))
    
    # Count word matches in source texts
    matches = 0
    total_source_words = 0
    
    for source in source_texts:
        source_words = set(re.findall(r'\b\w+\b', source.lower()))
        total_source_words += len(source_words)
        matches += len(generated_words.intersection(source_words))
    
    if total_source_words == 0:
        similarity = 0.0
    else:
        similarity = matches / max(len(generated_words), 1)
    
    # Potential hallucination if similarity is low
    is_potential_hallucination = similarity < threshold
    
    return {
        'similarity': similarity,
        'is_potential_hallucination': is_potential_hallucination,
        'generated_length': len(generated.split()),
        'unique_words': len(generated_words)
    }


def evaluate_model(model, dataloader, vocab: Dict[str, int], device: str, 
                  source_texts: List[str] = None, num_samples: int = 10) -> Dict:
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained language model
        dataloader: DataLoader for evaluation
        vocab: Vocabulary dictionary
        device: torch device
        source_texts: Source training texts for hallucination detection
        num_samples: Number of samples for generation evaluation
    Returns:
        Dictionary with all evaluation metrics
    """
    print("=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)
    
    # Calculate perplexity
    print("\n[1/3] Calculating Perplexity...")
    vocab_size = len(vocab)
    perplexity = calculate_perplexity(model, dataloader, device, vocab_size)
    print(f"Perplexity: {perplexity:.2f}")
    
    # Generate sample texts
    print("\n[2/3] Generating Sample Texts...")
    test_prompts = [
        "what is artificial intelligence",
        "how do neural networks work",
        "explain machine learning",
        "what is deep learning",
        "how does backpropagation work",
    ]
    
    generated_texts = []
    reference_texts = []
    
    for prompt in test_prompts[:num_samples]:
        generated = generate_text(model, vocab, prompt, max_length=30, temperature=0.8, device=device)
        generated_texts.append(generated)
        reference_texts.append(prompt)  # Simple reference
        
        print(f"  Prompt: '{prompt}'")
        print(f"  Generated: '{generated[:100]}...'")
        print()
    
    # Calculate BLEU (simplified)
    print("[3/3] Calculating BLEU Score...")
    bleu = calculate_bleu_score(generated_texts, reference_texts)
    print(f"BLEU Score: {bleu:.4f}")
    
    # Hallucination detection
    hallucination_results = []
    if source_texts:
        print("\n[4/4] Detecting Hallucinations...")
        sample_source = source_texts[:1000]  # Sample for efficiency
        for gen_text in generated_texts:
            hall_result = detect_hallucinations(gen_text, sample_source)
            hallucination_results.append(hall_result)
        
        avg_similarity = np.mean([r['similarity'] for r in hallucination_results])
        hallucination_rate = np.mean([r['is_potential_hallucination'] for r in hallucination_results])
        print(f"Average Similarity: {avg_similarity:.4f}")
        print(f"Potential Hallucination Rate: {hallucination_rate:.2%}")
    else:
        avg_similarity = None
        hallucination_rate = None
    
    results = {
        'perplexity': perplexity,
        'bleu_score': bleu,
        'generated_texts': generated_texts,
        'avg_similarity': avg_similarity,
        'hallucination_rate': hallucination_rate,
    }
    
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return results

