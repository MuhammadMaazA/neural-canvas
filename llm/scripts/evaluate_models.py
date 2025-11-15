"""
Model Evaluation and Benchmarking Script
=========================================
Evaluates Model 1 (custom) and Model 2 (pretrained) on multiple metrics:
- Perplexity (primary metric for language models)
- BLEU score (text generation quality)
- Response quality (length, diversity)
- Speed (inference time)

For coursework comparison and podcast presentation
"""

import os
os.environ['HF_HOME'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['HF_DATASETS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from tqdm import tqdm
import time
import json
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.art_expert_model import create_art_expert_model
from utils.curated_art_dataset import load_curated_art_datasets, TextDataset
from torch.utils.data import DataLoader, Subset


def load_model1(checkpoint_path: str, device: str):
    """Load custom model"""
    print(f"\nLoading Model 1 from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    tokenizer_name = checkpoint['config'].get('tokenizer_name', 'gpt2')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    model_size = checkpoint['config'].get('model_size', 'base')
    model = create_art_expert_model(tokenizer.vocab_size, model_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model 1 loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return model, tokenizer


def load_model2(checkpoint_path: str, device: str):
    """Load fine-tuned pretrained model"""
    print(f"\nLoading Model 2 from {checkpoint_path}...")

    # Try HF format first
    hf_path = checkpoint_path.replace('.pt', '_hf')
    if os.path.exists(hf_path):
        model = AutoModelForCausalLM.from_pretrained(hf_path)
        tokenizer = AutoTokenizer.from_pretrained(hf_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_name = checkpoint['config'].get('model_name', 'distilgpt2')
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])

    tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    model.eval()

    print(f"✓ Model 2 loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return model, tokenizer


@torch.no_grad()
def calculate_perplexity(model, dataloader, device, is_custom_model=True):
    """
    Calculate perplexity on validation set
    Lower is better (measures how "surprised" model is by the data)
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    for inputs, targets in tqdm(dataloader, desc="Calculating perplexity"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        if is_custom_model:
            # Custom model returns (logits, loss)
            _, loss = model(inputs, targets)
        else:
            # HuggingFace model
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss

        if loss is not None:
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')

    return perplexity, avg_loss


@torch.no_grad()
def measure_generation_quality(model, tokenizer, prompts: List[str], device):
    """
    Measure generation quality:
    - Average response length
    - Vocabulary diversity (unique tokens / total tokens)
    - Average generation time
    """
    model.eval()

    all_responses = []
    all_times = []
    all_tokens = []

    for prompt in tqdm(prompts, desc="Generating responses"):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs['input_ids'].to(device)

        # Measure generation time
        start_time = time.time()

        output_ids = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 100,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        gen_time = time.time() - start_time
        all_times.append(gen_time)

        # Decode response
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        all_responses.append(response)

        # Get tokens for diversity
        tokens = tokenizer.encode(response)
        all_tokens.extend(tokens)

    # Calculate metrics
    avg_length = np.mean([len(r.split()) for r in all_responses])
    vocab_diversity = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
    avg_time = np.mean(all_times)

    return {
        'avg_response_length': avg_length,
        'vocab_diversity': vocab_diversity,
        'avg_generation_time': avg_time,
        'samples': all_responses[:5]  # Save first 5 for inspection
    }


def create_test_prompts() -> List[str]:
    """Create diverse test prompts for evaluation"""
    prompts = [
        # Art knowledge
        "Question: What is Impressionism?\nAnswer:",
        "Question: Who was Pablo Picasso?\nAnswer:",
        "Question: Describe Baroque art.\nAnswer:",

        # AI literacy
        "Question: What is a neural network?\nAnswer:",
        "Question: How does machine learning work?\nAnswer:",
        "Question: Explain artificial intelligence to a beginner.\nAnswer:",

        # Integration
        "Question: How can AI recognize art styles?\nAnswer:",
        "Question: What is computer vision?\nAnswer:",
        "Question: Can computers understand beauty?\nAnswer:",

        # Conversational
        "User: Tell me about modern art.\nAssistant:",
        "User: I'm interested in learning about AI.\nAssistant:",
        "User: What makes a good painting?\nAssistant:",
    ]
    return prompts


def plot_comparison(results: Dict, save_path: str):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Perplexity comparison
    ax1 = axes[0, 0]
    models = list(results.keys())
    perplexities = [results[m]['perplexity'] for m in models]
    ax1.bar(models, perplexities, color=['steelblue', 'coral'])
    ax1.set_ylabel('Perplexity', fontsize=12)
    ax1.set_title('Perplexity (Lower is Better)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(perplexities):
        ax1.text(i, v + 1, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

    # Response length
    ax2 = axes[0, 1]
    lengths = [results[m]['generation_quality']['avg_response_length'] for m in models]
    ax2.bar(models, lengths, color=['steelblue', 'coral'])
    ax2.set_ylabel('Average Response Length (words)', fontsize=12)
    ax2.set_title('Response Length', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(lengths):
        ax2.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

    # Vocabulary diversity
    ax3 = axes[1, 0]
    diversity = [results[m]['generation_quality']['vocab_diversity'] for m in models]
    ax3.bar(models, diversity, color=['steelblue', 'coral'])
    ax3.set_ylabel('Vocabulary Diversity', fontsize=12)
    ax3.set_title('Unique Tokens / Total Tokens (Higher is Better)', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for i, v in enumerate(diversity):
        ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    # Generation speed
    ax4 = axes[1, 1]
    times = [results[m]['generation_quality']['avg_generation_time'] for m in models]
    ax4.bar(models, times, color=['steelblue', 'coral'])
    ax4.set_ylabel('Time (seconds)', fontsize=12)
    ax4.set_title('Average Generation Time (Lower is Faster)', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    for i, v in enumerate(times):
        ax4.text(i, v + 0.01, f'{v:.3f}s', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Comparison plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Art Expert Models")
    parser.add_argument(
        "--model1-path",
        type=str,
        default="/cs/student/projects1/2023/muhamaaz/checkpoints/model1_custom/best_model.pt",
        help="Path to Model 1 checkpoint"
    )
    parser.add_argument(
        "--model2-path",
        type=str,
        default="/cs/student/projects1/2023/muhamaaz/checkpoints/model2_pretrained/best_model.pt",
        help="Path to Model 2 checkpoint"
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=5000,
        help="Number of samples for perplexity evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/cs/student/projects1/2023/muhamaaz/evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("MODEL EVALUATION AND BENCHMARKING")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Eval samples: {args.eval_samples:,}")
    print("=" * 80 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load evaluation dataset
    print("\nLoading evaluation dataset...")
    all_texts, _ = load_curated_art_datasets(
        art_knowledge=10000,
        ai_literacy=10000,
        conversational=10000
    )

    # Load both models
    model1, tokenizer1 = load_model1(args.model1_path, args.device)
    model2, tokenizer2 = load_model2(args.model2_path, args.device)

    results = {}

    # =========================================================================
    # EVALUATE MODEL 1
    # =========================================================================
    print("\n" + "=" * 80)
    print("EVALUATING MODEL 1 (Custom)")
    print("=" * 80)

    # Prepare dataset
    eval_texts = all_texts[:args.eval_samples]
    dataset1 = TextDataset(eval_texts, tokenizer1, max_len=512)
    dataloader1 = DataLoader(dataset1, batch_size=8, shuffle=False, num_workers=2)

    # Perplexity
    print("\n[1/2] Calculating perplexity...")
    ppl1, loss1 = calculate_perplexity(model1, dataloader1, args.device, is_custom_model=True)
    print(f"✓ Perplexity: {ppl1:.2f} (Loss: {loss1:.4f})")

    # Generation quality
    print("\n[2/2] Measuring generation quality...")
    prompts = create_test_prompts()
    gen_quality1 = measure_generation_quality(model1, tokenizer1, prompts, args.device)
    print(f"✓ Avg response length: {gen_quality1['avg_response_length']:.1f} words")
    print(f"✓ Vocab diversity: {gen_quality1['vocab_diversity']:.3f}")
    print(f"✓ Avg generation time: {gen_quality1['avg_generation_time']:.3f}s")

    results['Model 1 (Custom)'] = {
        'perplexity': ppl1,
        'loss': loss1,
        'generation_quality': gen_quality1,
        'parameters': sum(p.numel() for p in model1.parameters()) / 1e6
    }

    # =========================================================================
    # EVALUATE MODEL 2
    # =========================================================================
    print("\n" + "=" * 80)
    print("EVALUATING MODEL 2 (Pretrained)")
    print("=" * 80)

    # Prepare dataset
    dataset2 = TextDataset(eval_texts, tokenizer2, max_len=512)
    dataloader2 = DataLoader(dataset2, batch_size=8, shuffle=False, num_workers=2)

    # Perplexity
    print("\n[1/2] Calculating perplexity...")
    ppl2, loss2 = calculate_perplexity(model2, dataloader2, args.device, is_custom_model=False)
    print(f"✓ Perplexity: {ppl2:.2f} (Loss: {loss2:.4f})")

    # Generation quality
    print("\n[2/2] Measuring generation quality...")
    gen_quality2 = measure_generation_quality(model2, tokenizer2, prompts, args.device)
    print(f"✓ Avg response length: {gen_quality2['avg_response_length']:.1f} words")
    print(f"✓ Vocab diversity: {gen_quality2['vocab_diversity']:.3f}")
    print(f"✓ Avg generation time: {gen_quality2['avg_generation_time']:.3f}s")

    results['Model 2 (Pretrained)'] = {
        'perplexity': ppl2,
        'loss': loss2,
        'generation_quality': gen_quality2,
        'parameters': sum(p.numel() for p in model2.parameters()) / 1e6
    }

    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS - COMPARISON")
    print("=" * 80)

    print("\n{:<25} {:>15} {:>15}".format("Metric", "Model 1 (Custom)", "Model 2 (Pretrained)"))
    print("-" * 80)
    print("{:<25} {:>15.2f} {:>15.2f}".format("Perplexity", ppl1, ppl2))
    print("{:<25} {:>15.1f} {:>15.1f}".format("Avg Response Length", gen_quality1['avg_response_length'], gen_quality2['avg_response_length']))
    print("{:<25} {:>15.3f} {:>15.3f}".format("Vocab Diversity", gen_quality1['vocab_diversity'], gen_quality2['vocab_diversity']))
    print("{:<25} {:>15.3f}s {:>15.3f}s".format("Avg Gen Time", gen_quality1['avg_generation_time'], gen_quality2['avg_generation_time']))
    print("{:<25} {:>15.1f}M {:>15.1f}M".format("Parameters", results['Model 1 (Custom)']['parameters'], results['Model 2 (Pretrained)']['parameters']))
    print("=" * 80)

    # Determine winner
    print("\nKEY INSIGHTS:")
    if ppl1 < ppl2:
        print(f"✓ Model 1 has LOWER perplexity ({ppl1:.2f} vs {ppl2:.2f}) - better language modeling")
    else:
        print(f"✓ Model 2 has LOWER perplexity ({ppl2:.2f} vs {ppl1:.2f}) - better language modeling")

    if gen_quality1['vocab_diversity'] > gen_quality2['vocab_diversity']:
        print(f"✓ Model 1 has MORE diverse vocabulary ({gen_quality1['vocab_diversity']:.3f} vs {gen_quality2['vocab_diversity']:.3f})")
    else:
        print(f"✓ Model 2 has MORE diverse vocabulary ({gen_quality2['vocab_diversity']:.3f} vs {gen_quality1['vocab_diversity']:.3f})")

    if gen_quality1['avg_generation_time'] < gen_quality2['avg_generation_time']:
        print(f"✓ Model 1 is FASTER ({gen_quality1['avg_generation_time']:.3f}s vs {gen_quality2['avg_generation_time']:.3f}s)")
    else:
        print(f"✓ Model 2 is FASTER ({gen_quality2['avg_generation_time']:.3f}s vs {gen_quality1['avg_generation_time']:.3f}s)")

    # Save results
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        # Convert to serializable format
        serializable_results = {
            k: {
                'perplexity': float(v['perplexity']),
                'loss': float(v['loss']),
                'parameters': float(v['parameters']),
                'avg_response_length': float(v['generation_quality']['avg_response_length']),
                'vocab_diversity': float(v['generation_quality']['vocab_diversity']),
                'avg_generation_time': float(v['generation_quality']['avg_generation_time']),
                'sample_responses': v['generation_quality']['samples']
            }
            for k, v in results.items()
        }
        json.dump(serializable_results, f, indent=2)
    print(f"\n✓ Results saved to: {results_file}")

    # Create comparison plot
    plot_path = os.path.join(args.output_dir, "model_comparison.png")
    plot_comparison(results, plot_path)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
