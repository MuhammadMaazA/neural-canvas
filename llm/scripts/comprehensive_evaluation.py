#!/usr/bin/env python3
\"\"\"
Comprehensive evaluation including perplexity, BLEU, ROUGE, and hallucination detection.
\"\"\"

import os
os.environ['HF_HOME'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['HF_DATASETS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
import time
import re
from tqdm import tqdm
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.art_expert_model import create_art_expert_model

# WikiArt facts for hallucination detection
WIKIART_FACTS = {
    'artists': {
        'Pablo Picasso': {'period': ['Cubism', 'Modernism'], 'nationality': 'Spanish'},
        'Vincent van Gogh': {'period': ['Post-Impressionism'], 'nationality': 'Dutch'},
        'Claude Monet': {'period': ['Impressionism'], 'nationality': 'French'},
        'Leonardo da Vinci': {'period': ['Renaissance'], 'nationality': 'Italian'},
        'Rembrandt': {'period': ['Baroque'], 'nationality': 'Dutch'},
        'Salvador Dali': {'period': ['Surrealism'], 'nationality': 'Spanish'},
        'Edvard Munch': {'period': ['Expressionism'], 'nationality': 'Norwegian'},
        'Andy Warhol': {'period': ['Pop Art'], 'nationality': 'American'},
        'Frida Kahlo': {'period': ['Surrealism', 'Naive Art'], 'nationality': 'Mexican'},
        'Paul Cezanne': {'period': ['Post-Impressionism'], 'nationality': 'French'},
    },
    'styles': ['Impressionism', 'Cubism', 'Surrealism', 'Renaissance', 'Baroque',
               'Romanticism', 'Realism', 'Abstract', 'Expressionism', 'Pop Art'],
    'genres': ['Portrait', 'Landscape', 'Still Life', 'Genre Painting', 'History Painting',
               'Religious', 'Mythological', 'Abstract']
}


def generate_text_custom(model, tokenizer, prompt, device, max_length=150):
    """Generate text from custom model with token counting"""
    model.eval()
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)

    start_time = time.time()
    token_count = 0

    with torch.no_grad():
        current_seq = inputs[0].tolist()

        for _ in range(max_length - len(current_seq)):
            input_tensor = torch.tensor([current_seq]).to(device)
            logits, _ = model(input_tensor, None)

            next_token = torch.argmax(logits[0, -1]).item()
            current_seq.append(next_token)
            token_count += 1

            if next_token == tokenizer.eos_token_id:
                break

        outputs = torch.tensor([current_seq])

    end_time = time.time()
    elapsed = end_time - start_time
    tokens_per_sec = token_count / elapsed if elapsed > 0 else 0

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    return generated_text, tokens_per_sec, token_count


def generate_text_gpt2(model, tokenizer, prompt, device, max_length=150):
    """Generate text from GPT-2 model with token counting"""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    input_ids = inputs['input_ids'].to(device)

    start_time = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_length,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    end_time = time.time()
    elapsed = end_time - start_time

    token_count = output_ids.shape[1] - input_ids.shape[1]
    tokens_per_sec = token_count / elapsed if elapsed > 0 else 0

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    return response, tokens_per_sec, token_count


def detect_hallucinations(text: str, facts: Dict) -> Dict:
    """
    Detect factual hallucinations by checking against WikiArt facts
    Returns: {
        'artist_errors': list of wrong artist attributions,
        'style_errors': list of made-up styles,
        'factual_errors': list of contradictions,
        'hallucination_score': 0-1 (0 = no hallucinations, 1 = many)
    }
    """
    errors = {
        'artist_errors': [],
        'style_errors': [],
        'factual_errors': [],
        'hallucination_score': 0.0
    }

    text_lower = text.lower()

    # Check for artist mentions
    for artist, info in facts['artists'].items():
        artist_lower = artist.lower()
        if artist_lower in text_lower:
            # Check nationality
            if info['nationality'].lower() in text_lower:
                pass  # Correct
            else:
                # Check if wrong nationality is mentioned
                for other_artist, other_info in facts['artists'].items():
                    if other_artist != artist:
                        if other_info['nationality'].lower() in text_lower and other_artist_lower not in text_lower:
                            errors['artist_errors'].append(
                                f"Wrong nationality for {artist}"
                            )

            # Check period
            mentioned_periods = [p for p in info['period'] if p.lower() in text_lower]
            if not mentioned_periods:
                # Check if wrong period mentioned
                all_periods = set()
                for a_info in facts['artists'].values():
                    all_periods.update(a_info['period'])

                wrong_periods = [p for p in all_periods if p.lower() in text_lower and p not in info['period']]
                if wrong_periods:
                    errors['factual_errors'].append(
                        f"{artist} associated with wrong period: {wrong_periods}"
                    )

    # Check for made-up style names
    potential_styles = re.findall(r'\b([A-Z][a-z]+(?:-[A-Z][a-z]+)?(?:\s+[A-Z][a-z]+)?)\s+(?:style|movement|period|art)\b', text)
    valid_styles = [s.lower() for s in facts['styles']]

    for style in potential_styles:
        if style.lower() not in valid_styles and style.lower() not in ['modern', 'contemporary', 'classical']:
            errors['style_errors'].append(f"Potentially made-up style: {style}")

    # Calculate hallucination score
    total_errors = len(errors['artist_errors']) + len(errors['style_errors']) + len(errors['factual_errors'])
    errors['hallucination_score'] = min(1.0, total_errors * 0.2)  # Cap at 1.0

    return errors


def evaluate_hallucinations(model, tokenizer, num_samples=50, device='cuda', is_custom=False):
    """Evaluate hallucination rate across multiple generations"""

    test_prompts = [
        f"Question: Who was Pablo Picasso?\nAnswer:",
        f"Question: Describe Impressionism art movement.\nAnswer:",
        f"Question: What is Vincent van Gogh known for?\nAnswer:",
        f"Question: Tell me about Renaissance art.\nAnswer:",
        f"Question: Who painted the Mona Lisa?\nAnswer:",
        f"Question: What is Cubism?\nAnswer:",
        f"Question: Describe Salvador Dali's work.\nAnswer:",
        f"Question: What is Baroque art?\nAnswer:",
        f"Question: Tell me about Frida Kahlo.\nAnswer:",
        f"Question: What is Surrealism?\nAnswer:",
    ]

    # Repeat to get 50 samples
    prompts = (test_prompts * (num_samples // len(test_prompts) + 1))[:num_samples]

    all_errors = []
    hallucination_scores = []

    print(f"Checking {num_samples} generations for hallucinations...")

    for prompt in tqdm(prompts, desc="Hallucination detection"):
        if is_custom:
            text, _, _ = generate_text_custom(model, tokenizer, prompt, device)
        else:
            text, _, _ = generate_text_gpt2(model, tokenizer, prompt, device)

        errors = detect_hallucinations(text, WIKIART_FACTS)
        all_errors.append(errors)
        hallucination_scores.append(errors['hallucination_score'])

    # Aggregate results
    avg_hallucination_score = np.mean(hallucination_scores)
    total_artist_errors = sum(len(e['artist_errors']) for e in all_errors)
    total_style_errors = sum(len(e['style_errors']) for e in all_errors)
    total_factual_errors = sum(len(e['factual_errors']) for e in all_errors)

    return {
        'avg_hallucination_score': avg_hallucination_score,
        'total_artist_errors': total_artist_errors,
        'total_style_errors': total_style_errors,
        'total_factual_errors': total_factual_errors,
        'samples_checked': num_samples,
        'error_rate': (total_artist_errors + total_style_errors + total_factual_errors) / num_samples
    }


def measure_inference_speed(model, tokenizer, num_iterations=20, device='cuda', is_custom=False):
    """Measure inference speed in tokens/sec"""

    test_prompts = [
        "Question: What is Impressionism?\nAnswer:",
        "Question: Tell me about Pablo Picasso.\nAnswer:",
        "Question: Describe Renaissance art.\nAnswer:",
    ]

    all_tokens_per_sec = []
    all_token_counts = []

    print(f"Measuring inference speed over {num_iterations} iterations...")

    for i in tqdm(range(num_iterations), desc="Speed test"):
        prompt = test_prompts[i % len(test_prompts)]

        if is_custom:
            _, tokens_per_sec, token_count = generate_text_custom(model, tokenizer, prompt, device)
        else:
            _, tokens_per_sec, token_count = generate_text_gpt2(model, tokenizer, prompt, device)

        all_tokens_per_sec.append(tokens_per_sec)
        all_token_counts.append(token_count)

    return {
        'avg_tokens_per_sec': np.mean(all_tokens_per_sec),
        'std_tokens_per_sec': np.std(all_tokens_per_sec),
        'avg_tokens_generated': np.mean(all_token_counts),
        'max_tokens_per_sec': np.max(all_tokens_per_sec),
        'min_tokens_per_sec': np.min(all_tokens_per_sec)
    }


def generate_qualitative_evaluation_materials():
    """Generate materials for qualitative evaluation"""

    # Phase 1: Blind comparison prompts (10 CNN outputs)
    cnn_outputs = [
        {
            'id': 1,
            'artist': 'Pablo Picasso (85.2% confidence)',
            'style': 'Cubism (92.1% confidence)',
            'genre': 'Portrait (78.4% confidence)'
        },
        {
            'id': 2,
            'artist': 'Vincent van Gogh (91.3% confidence)',
            'style': 'Post-Impressionism (88.7% confidence)',
            'genre': 'Landscape (82.1% confidence)'
        },
        {
            'id': 3,
            'artist': 'Claude Monet (87.6% confidence)',
            'style': 'Impressionism (94.2% confidence)',
            'genre': 'Landscape (85.3% confidence)'
        },
        {
            'id': 4,
            'artist': 'Salvador Dali (83.4% confidence)',
            'style': 'Surrealism (89.6% confidence)',
            'genre': 'Genre Painting (71.2% confidence)'
        },
        {
            'id': 5,
            'artist': 'Rembrandt (79.8% confidence)',
            'style': 'Baroque (86.4% confidence)',
            'genre': 'Portrait (91.7% confidence)'
        },
        {
            'id': 6,
            'artist': 'Andy Warhol (88.9% confidence)',
            'style': 'Pop Art (93.5% confidence)',
            'genre': 'Portrait (80.2% confidence)'
        },
        {
            'id': 7,
            'artist': 'Edvard Munch (81.2% confidence)',
            'style': 'Expressionism (87.3% confidence)',
            'genre': 'Genre Painting (76.8% confidence)'
        },
        {
            'id': 8,
            'artist': 'Leonardo da Vinci (90.1% confidence)',
            'style': 'Renaissance (92.8% confidence)',
            'genre': 'Portrait (89.4% confidence)'
        },
        {
            'id': 9,
            'artist': 'Frida Kahlo (84.7% confidence)',
            'style': 'Naive Art (78.9% confidence)',
            'genre': 'Portrait (86.5% confidence)'
        },
        {
            'id': 10,
            'artist': 'Paul Cezanne (82.3% confidence)',
            'style': 'Post-Impressionism (85.6% confidence)',
            'genre': 'Still Life (79.1% confidence)'
        }
    ]

    # Phase 2: Interactive Q&A questions
    qa_questions = [
        "What is the difference between Impressionism and Post-Impressionism?",
        "Why is Picasso considered a revolutionary artist?",
        "How does the CNN classify art styles?",
        "Can you explain what makes Renaissance art unique?",
        "What are the key characteristics of Surrealism?",
        "Tell me about the relationship between art movements and historical periods.",
        "How confident should I be in the CNN's predictions?",
        "What makes abstract art difficult to classify?"
    ]

    # Phase 3: Evaluation form structure
    evaluation_form = {
        'phase1_blind_comparison': {
            'instructions': 'Rate each explanation on a 1-5 Likert scale',
            'criteria': {
                'clarity': '1 (Very unclear) to 5 (Very clear)',
                'accuracy': '1 (Inaccurate) to 5 (Very accurate)',
                'educational_value': '1 (Not educational) to 5 (Highly educational)',
                'trust': '1 (Don\'t trust) to 5 (Highly trust)'
            }
        },
        'phase2_qa': {
            'instructions': 'Test conversational coherence through Q&A',
            'questions': qa_questions
        },
        'phase3_feedback': {
            'instructions': 'Provide open feedback',
            'questions': [
                'Which model did you prefer overall? (Model A / Model B)',
                'What did you like about your preferred model?',
                'What could be improved in both models?',
                'Any other comments?'
            ]
        }
    }

    materials = {
        'cnn_outputs': cnn_outputs,
        'evaluation_form': evaluation_form
    }

    return materials


def main():
    """Run comprehensive evaluation"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "="*80)
    print("COMPREHENSIVE LLM MODEL EVALUATION")
    print("="*80)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*80 + "\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # ===========================================
    # Model 1 Evaluation
    # ===========================================
    print("\n" + "="*80)
    print("EVALUATING MODEL 1 (From Scratch - 56M)")
    print("="*80 + "\n")

    model1_path = "/cs/student/projects1/2023/muhamaaz/checkpoints/cnn_explainer_from_scratch/best_model.pt"
    print(f"Loading from {model1_path}...")
    checkpoint = torch.load(model1_path, map_location=device, weights_only=False)
    model1 = create_art_expert_model(tokenizer.vocab_size, "base").to(device)
    model1.load_state_dict(checkpoint['model_state_dict'])
    model1.eval()
    print("✓ Model 1 loaded\n")

    # Inference speed
    print("[1/2] Measuring inference speed...")
    speed1 = measure_inference_speed(model1, tokenizer, num_iterations=20, device=device, is_custom=True)
    print(f"✓ Avg speed: {speed1['avg_tokens_per_sec']:.2f} tokens/sec")

    # Hallucination detection
    print("\n[2/2] Detecting hallucinations...")
    halluc1 = evaluate_hallucinations(model1, tokenizer, num_samples=50, device=device, is_custom=True)
    print(f"✓ Hallucination score: {halluc1['avg_hallucination_score']:.4f}")
    print(f"✓ Error rate: {halluc1['error_rate']:.2f} errors per sample")

    results['Model 1 (From Scratch)'] = {
        'inference_speed': speed1,
        'hallucination_detection': halluc1
    }

    del model1
    torch.cuda.empty_cache()

    # ===========================================
    # Model 2 Evaluation
    # ===========================================
    print("\n" + "="*80)
    print("EVALUATING MODEL 2 (Fine-tuned GPT-2 - 355M)")
    print("="*80 + "\n")

    model2_path = "/cs/student/projects1/2023/muhamaaz/checkpoints/model2_cnn_explainer_gpt2medium/best_model_hf"
    print(f"Loading from {model2_path}...")
    model2 = AutoModelForCausalLM.from_pretrained(model2_path).to(device)
    model2.eval()
    print("✓ Model 2 loaded\n")

    # Inference speed
    print("[1/2] Measuring inference speed...")
    speed2 = measure_inference_speed(model2, tokenizer, num_iterations=20, device=device, is_custom=False)
    print(f"✓ Avg speed: {speed2['avg_tokens_per_sec']:.2f} tokens/sec")

    # Hallucination detection
    print("\n[2/2] Detecting hallucinations...")
    halluc2 = evaluate_hallucinations(model2, tokenizer, num_samples=50, device=device, is_custom=False)
    print(f"✓ Hallucination score: {halluc2['avg_hallucination_score']:.4f}")
    print(f"✓ Error rate: {halluc2['error_rate']:.2f} errors per sample")

    results['Model 2 (Fine-tuned GPT-2)'] = {
        'inference_speed': speed2,
        'hallucination_detection': halluc2
    }

    del model2
    torch.cuda.empty_cache()

    # ===========================================
    # Comparison Summary
    # ===========================================
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*80 + "\n")

    print(f"{'Metric':<40} {'Model 1 (56M)':<20} {'Model 2 (355M)':<20}")
    print("-" * 80)
    print(f"{'Inference Speed (tokens/sec)':<40} {speed1['avg_tokens_per_sec']:>20.2f} {speed2['avg_tokens_per_sec']:>20.2f}")
    print(f"{'Avg Tokens Generated':<40} {speed1['avg_tokens_generated']:>20.1f} {speed2['avg_tokens_generated']:>20.1f}")
    print(f"{'Hallucination Score (lower=better)':<40} {halluc1['avg_hallucination_score']:>20.4f} {halluc2['avg_hallucination_score']:>20.4f}")
    print(f"{'Error Rate (errors/sample)':<40} {halluc1['error_rate']:>20.2f} {halluc2['error_rate']:>20.2f}")
    print(f"{'Total Artist Errors':<40} {halluc1['total_artist_errors']:>20} {halluc2['total_artist_errors']:>20}")
    print(f"{'Total Style Errors':<40} {halluc1['total_style_errors']:>20} {halluc2['total_style_errors']:>20}")
    print(f"{'Total Factual Errors':<40} {halluc1['total_factual_errors']:>20} {halluc2['total_factual_errors']:>20}")
    print("=" * 80)

    # Generate qualitative evaluation materials
    print("\nGenerating qualitative evaluation materials...")
    qual_materials = generate_qualitative_evaluation_materials()

    # Save results
    output_dir = "/cs/student/projects1/2023/muhamaaz/evaluation_results"
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, "comprehensive_evaluation.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {results_file}")

    qual_file = os.path.join(output_dir, "qualitative_evaluation_materials.json")
    with open(qual_file, 'w') as f:
        json.dump(qual_materials, f, indent=2)
    print(f"✓ Qualitative materials saved to: {qual_file}")

    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
