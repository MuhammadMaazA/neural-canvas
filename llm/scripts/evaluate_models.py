\"\"\"
Evaluation script for LLM models.
Computes perplexity, BLEU, ROUGE, and other metrics.
\"\"\"

import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LLM_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = LLM_DIR.parent

if os.path.exists('/cs/student/projects1/2023/muhamaaz/datasets'):
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
from collections import defaultdict

import sys
sys.path.insert(0, str(LLM_DIR))

from models.art_expert_model import create_art_expert_model
from utils.clean_art_critic_dataset import load_clean_art_critic_dataset
from torch.utils.data import DataLoader, Subset

# Try to import evaluation metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    NLTK_AVAILABLE = True
except:
    print("⚠ NLTK not available. BLEU scores will be skipped.")
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except:
    print("⚠ rouge-score not available. ROUGE scores will be skipped.")
    ROUGE_AVAILABLE = False


def load_model1(checkpoint_path: str, device: str):
    """Load custom model (from scratch)"""
    print(f"\nLoading Model 1 (From Scratch) from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    tokenizer_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    model_size = checkpoint['config'].get('model_size', 'base')
    model = create_art_expert_model(tokenizer.vocab_size, model_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model 1 loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return model, tokenizer


def load_model2(model_path: str, device: str):
    """Load fine-tuned GPT-2 model"""
    print(f"\nLoading Model 2 (Fine-tuned GPT-2) from {model_path}...")

    # Load from Hugging Face format
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

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


def generate_text_custom(model, tokenizer, prompt, device, max_length=150):
    """Generate text from custom model"""
    model.eval()
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        current_seq = inputs[0].tolist()

        for _ in range(max_length - len(current_seq)):
            input_tensor = torch.tensor([current_seq]).to(device)
            logits, _ = model(input_tensor, None)

            next_token = torch.argmax(logits[0, -1]).item()
            current_seq.append(next_token)

            if next_token == tokenizer.eos_token_id:
                break

        outputs = torch.tensor([current_seq])

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    return generated_text


@torch.no_grad()
def measure_generation_quality(model, tokenizer, prompts: List[str], device, is_custom=False):
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
        # Measure generation time
        start_time = time.time()

        if is_custom:
            response = generate_text_custom(model, tokenizer, prompt, device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            input_ids = inputs['input_ids'].to(device)

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

            # Decode response
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()

        gen_time = time.time() - start_time
        all_times.append(gen_time)
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
    }, all_responses


def calculate_nlp_metrics(references: List[str], hypotheses: List[str]):
    """Calculate BLEU, ROUGE, and METEOR scores"""
    metrics = {}

    # BLEU Scores
    if NLTK_AVAILABLE:
        smoothing = SmoothingFunction().method1
        bleu_scores = []

        for ref, hyp in zip(references, hypotheses):
            ref_tokens = ref.split()
            hyp_tokens = hyp.split()

            bleu_1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu_2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu_3 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            bleu_4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

            bleu_scores.append({
                'bleu-1': bleu_1,
                'bleu-2': bleu_2,
                'bleu-3': bleu_3,
                'bleu-4': bleu_4
            })

        metrics['bleu-1'] = np.mean([s['bleu-1'] for s in bleu_scores])
        metrics['bleu-2'] = np.mean([s['bleu-2'] for s in bleu_scores])
        metrics['bleu-3'] = np.mean([s['bleu-3'] for s in bleu_scores])
        metrics['bleu-4'] = np.mean([s['bleu-4'] for s in bleu_scores])

    # ROUGE Scores
    if ROUGE_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = defaultdict(list)

        for ref, hyp in zip(references, hypotheses):
            scores = scorer.score(ref, hyp)
            for metric_name, score_obj in scores.items():
                rouge_scores[metric_name].append(score_obj.fmeasure)

        metrics['rouge-1'] = np.mean(rouge_scores['rouge1'])
        metrics['rouge-2'] = np.mean(rouge_scores['rouge2'])
        metrics['rouge-L'] = np.mean(rouge_scores['rougeL'])

    return metrics


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


def get_default_path(name):
    """Get path - prefer local, fallback to server"""
    local_path = PROJECT_ROOT / name
    if local_path.exists():
        return str(local_path)
    return f"/cs/student/projects1/2023/muhamaaz/{name}"

def main():
    # Set default paths based on environment
    default_model1 = get_default_path("checkpoints/cnn_explainer_from_scratch/best_model.pt")
    default_model2 = get_default_path("checkpoints/model2_cnn_explainer_gpt2medium/best_model_hf")
    default_output = get_default_path("evaluation_results")
    
    parser = argparse.ArgumentParser(description="Evaluate LLM Models")
    parser.add_argument(
        "--model1-path",
        type=str,
        default=default_model1,
        help="Path to Model 1 checkpoint (from scratch)"
    )
    parser.add_argument(
        "--model2-path",
        type=str,
        default=default_model2,
        help="Path to Model 2 checkpoint (fine-tuned GPT-2)"
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=500,
        help="Number of samples for evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output,
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
    print("LLM MODEL EVALUATION AND BENCHMARKING")
    print("=" * 80)
    print(f"Device: {args.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Eval samples: {args.eval_samples:,}")
    print("=" * 80 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Load evaluation dataset
    print("Loading evaluation dataset...")
    full_dataset = load_clean_art_critic_dataset(
        tokenizer=tokenizer,
        max_len=512,
        size="small"  # Use small for faster evaluation
    )

    # Load both models
    model1, tokenizer1 = load_model1(args.model1_path, args.device)
    model2, tokenizer2 = load_model2(args.model2_path, args.device)

    results = {}

    # =========================================================================
    # EVALUATE MODEL 1 (From Scratch)
    # =========================================================================
    print("\n" + "=" * 80)
    print("EVALUATING MODEL 1 (From Scratch - 56M params)")
    print("=" * 80)

    # Prepare dataset
    eval_subset = Subset(full_dataset, range(min(args.eval_samples, len(full_dataset))))
    dataloader1 = DataLoader(eval_subset, batch_size=8, shuffle=False, num_workers=2)

    # [1/3] Perplexity
    print("\n[1/3] Calculating perplexity...")
    ppl1, loss1 = calculate_perplexity(model1, dataloader1, args.device, is_custom_model=True)
    print(f"✓ Perplexity: {ppl1:.2f} (Loss: {loss1:.4f})")

    # [2/3] Generation quality
    print("\n[2/3] Measuring generation quality...")
    prompts = create_test_prompts()
    gen_quality1, generated_texts1 = measure_generation_quality(model1, tokenizer1, prompts, args.device, is_custom=True)
    print(f"✓ Avg response length: {gen_quality1['avg_response_length']:.1f} words")
    print(f"✓ Vocab diversity: {gen_quality1['vocab_diversity']:.3f}")
    print(f"✓ Avg generation time: {gen_quality1['avg_generation_time']:.3f}s")

    # [3/3] NLP Metrics (BLEU, ROUGE)
    print("\n[3/3] Calculating NLP metrics (BLEU, ROUGE)...")
    # Use prompts as references for this simple test
    nlp_metrics1 = calculate_nlp_metrics(prompts, generated_texts1)
    if NLTK_AVAILABLE:
        print(f"✓ BLEU-4: {nlp_metrics1.get('bleu-4', 0):.4f}")
    if ROUGE_AVAILABLE:
        print(f"✓ ROUGE-L: {nlp_metrics1.get('rouge-L', 0):.4f}")

    results['Model 1 (From Scratch)'] = {
        'perplexity': ppl1,
        'loss': loss1,
        'generation_quality': gen_quality1,
        'nlp_metrics': nlp_metrics1,
        'parameters': sum(p.numel() for p in model1.parameters()) / 1e6
    }

    # =========================================================================
    # EVALUATE MODEL 2 (Fine-tuned GPT-2)
    # =========================================================================
    print("\n" + "=" * 80)
    print("EVALUATING MODEL 2 (Fine-tuned GPT-2 Medium - 355M params)")
    print("=" * 80)

    # Prepare dataset
    dataloader2 = DataLoader(eval_subset, batch_size=8, shuffle=False, num_workers=2)

    # [1/3] Perplexity
    print("\n[1/3] Calculating perplexity...")
    ppl2, loss2 = calculate_perplexity(model2, dataloader2, args.device, is_custom_model=False)
    print(f"✓ Perplexity: {ppl2:.2f} (Loss: {loss2:.4f})")

    # [2/3] Generation quality
    print("\n[2/3] Measuring generation quality...")
    gen_quality2, generated_texts2 = measure_generation_quality(model2, tokenizer2, prompts, args.device, is_custom=False)
    print(f"✓ Avg response length: {gen_quality2['avg_response_length']:.1f} words")
    print(f"✓ Vocab diversity: {gen_quality2['vocab_diversity']:.3f}")
    print(f"✓ Avg generation time: {gen_quality2['avg_generation_time']:.3f}s")

    # [3/3] NLP Metrics (BLEU, ROUGE)
    print("\n[3/3] Calculating NLP metrics (BLEU, ROUGE)...")
    nlp_metrics2 = calculate_nlp_metrics(prompts, generated_texts2)
    if NLTK_AVAILABLE:
        print(f"✓ BLEU-4: {nlp_metrics2.get('bleu-4', 0):.4f}")
    if ROUGE_AVAILABLE:
        print(f"✓ ROUGE-L: {nlp_metrics2.get('rouge-L', 0):.4f}")

    results['Model 2 (Fine-tuned GPT-2)'] = {
        'perplexity': ppl2,
        'loss': loss2,
        'generation_quality': gen_quality2,
        'nlp_metrics': nlp_metrics2,
        'parameters': sum(p.numel() for p in model2.parameters()) / 1e6
    }

    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS - COMPARISON")
    print("=" * 80)

    print("\n{:<25} {:>20} {:>20}".format("Metric", "Model 1 (56M)", "Model 2 (355M)"))
    print("-" * 80)
    print("{:<25} {:>20.2f} {:>20.2f}".format("Perplexity", ppl1, ppl2))
    print("{:<25} {:>20.4f} {:>20.4f}".format("Loss", loss1, loss2))

    if NLTK_AVAILABLE:
        print("{:<25} {:>20.4f} {:>20.4f}".format("BLEU-1", nlp_metrics1.get('bleu-1', 0), nlp_metrics2.get('bleu-1', 0)))
        print("{:<25} {:>20.4f} {:>20.4f}".format("BLEU-2", nlp_metrics1.get('bleu-2', 0), nlp_metrics2.get('bleu-2', 0)))
        print("{:<25} {:>20.4f} {:>20.4f}".format("BLEU-3", nlp_metrics1.get('bleu-3', 0), nlp_metrics2.get('bleu-3', 0)))
        print("{:<25} {:>20.4f} {:>20.4f}".format("BLEU-4", nlp_metrics1.get('bleu-4', 0), nlp_metrics2.get('bleu-4', 0)))

    if ROUGE_AVAILABLE:
        print("{:<25} {:>20.4f} {:>20.4f}".format("ROUGE-1", nlp_metrics1.get('rouge-1', 0), nlp_metrics2.get('rouge-1', 0)))
        print("{:<25} {:>20.4f} {:>20.4f}".format("ROUGE-2", nlp_metrics1.get('rouge-2', 0), nlp_metrics2.get('rouge-2', 0)))
        print("{:<25} {:>20.4f} {:>20.4f}".format("ROUGE-L", nlp_metrics1.get('rouge-L', 0), nlp_metrics2.get('rouge-L', 0)))

    print("{:<25} {:>20.1f} {:>20.1f}".format("Avg Response Length", gen_quality1['avg_response_length'], gen_quality2['avg_response_length']))
    print("{:<25} {:>20.3f} {:>20.3f}".format("Vocab Diversity", gen_quality1['vocab_diversity'], gen_quality2['vocab_diversity']))
    print("{:<25} {:>17.3f}s {:>17.3f}s".format("Avg Gen Time", gen_quality1['avg_generation_time'], gen_quality2['avg_generation_time']))
    print("=" * 80)

    # Determine winner
    print("\nKEY INSIGHTS:")
    if ppl1 < ppl2:
        print(f"✓ Model 1 has LOWER perplexity ({ppl1:.2f} vs {ppl2:.2f}) - better language modeling")
    else:
        print(f"✓ Model 2 has LOWER perplexity ({ppl2:.2f} vs {ppl1:.2f}) - better language modeling")

    if NLTK_AVAILABLE:
        bleu4_1 = nlp_metrics1.get('bleu-4', 0)
        bleu4_2 = nlp_metrics2.get('bleu-4', 0)
        if bleu4_1 > bleu4_2:
            print(f"✓ Model 1 has HIGHER BLEU-4 score ({bleu4_1:.4f} vs {bleu4_2:.4f}) - better generation quality")
        else:
            print(f"✓ Model 2 has HIGHER BLEU-4 score ({bleu4_2:.4f} vs {bleu4_1:.4f}) - better generation quality")

    if ROUGE_AVAILABLE:
        rougeL_1 = nlp_metrics1.get('rouge-L', 0)
        rougeL_2 = nlp_metrics2.get('rouge-L', 0)
        if rougeL_1 > rougeL_2:
            print(f"✓ Model 1 has HIGHER ROUGE-L score ({rougeL_1:.4f} vs {rougeL_2:.4f})")
        else:
            print(f"✓ Model 2 has HIGHER ROUGE-L score ({rougeL_2:.4f} vs {rougeL_1:.4f})")

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
        serializable_results = {}
        for k, v in results.items():
            serializable_results[k] = {
                'perplexity': float(v['perplexity']),
                'loss': float(v['loss']),
                'parameters': float(v['parameters']),
                'avg_response_length': float(v['generation_quality']['avg_response_length']),
                'vocab_diversity': float(v['generation_quality']['vocab_diversity']),
                'avg_generation_time': float(v['generation_quality']['avg_generation_time']),
                'sample_responses': v['generation_quality']['samples']
            }
            # Add NLP metrics if available
            if 'nlp_metrics' in v:
                for metric_name, metric_value in v['nlp_metrics'].items():
                    serializable_results[k][metric_name] = float(metric_value)

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
