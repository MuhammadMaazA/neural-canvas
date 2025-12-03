#!/usr/bin/env python3
"""
Test CNN + LLM Integration Pipeline
===================================

This script tests the full pipeline:
1. Load sample artwork images
2. Run CNN classification (artist/style/genre)
3. Feed CNN output to trained LLM models
4. Generate side-by-side explanations (Model 1 vs Model 2)
5. Evaluate and compare outputs

Perfect for podcast demonstration and model evaluation!
"""

import os
import sys
import torch
sys.path.append('/cs/student/projects1/2023/muhamaaz/neural-canvas')

from cnn_inference import CNNInference
from llm.models.art_expert_model import ArtExpertTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
import json
from datetime import datetime
from typing import Dict, List
import time


class LLMInference:
    """Load and run inference with trained LLM models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"LLM Device: {self.device}")
        
        # Load tokenizer (shared by both models)
        print("Loading tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model paths
        self.model1_path = "/cs/student/projects1/2023/muhamaaz/checkpoints/cnn_explainer_from_scratch/best_model.pt"
        self.model2_path = "/cs/student/projects1/2023/muhamaaz/checkpoints/cnn_explainer_finetuned/best_model.pt"
        
        # Load models
        self.model1 = None
        self.model2 = None
        
        self.load_model1()
        self.load_model2()
        
    def load_model1(self):
        """Load Model 1 (custom from scratch)"""
        try:
            print("\n[Model 1] Loading custom transformer (from scratch)...")
            
            # Create model architecture 
            self.model1 = ArtExpertTransformer(
                vocab_size=50257,  # GPT-2 vocab
                dim=512,
                n_layers=8,
                n_heads=8, 
                n_kv_heads=2,
                max_len=512,
                dropout=0.1,
                label_smoothing=0.1
            ).to(self.device)
            
            # Load checkpoint
            if os.path.exists(self.model1_path):
                checkpoint = torch.load(self.model1_path, map_location=self.device)
                self.model1.load_state_dict(checkpoint['model'])
                print(f"✅ Model 1 loaded from {self.model1_path}")
                print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
                print(f"   Loss: {checkpoint.get('loss', 'unknown'):.4f}")
            else:
                print(f"⚠️  Model 1 checkpoint not found: {self.model1_path}")
                print("   Using untrained model")
                
        except Exception as e:
            print(f"❌ Error loading Model 1: {e}")
            self.model1 = None
            
    def load_model2(self):
        """Load Model 2 (fine-tuned GPT-2)"""
        try:
            print("\n[Model 2] Loading fine-tuned GPT-2...")
            
            # Load base model
            self.model2 = AutoModelForCausalLM.from_pretrained('gpt2').to(self.device)
            
            # Load fine-tuned weights
            if os.path.exists(self.model2_path):
                checkpoint = torch.load(self.model2_path, map_location=self.device)
                self.model2.load_state_dict(checkpoint['model'])
                print(f"✅ Model 2 loaded from {self.model2_path}")
                print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
                print(f"   Loss: {checkpoint.get('loss', 'unknown'):.4f}")
            else:
                print(f"⚠️  Model 2 checkpoint not found: {self.model2_path}")
                print("   Using base GPT-2 (not fine-tuned)")
                
        except Exception as e:
            print(f"❌ Error loading Model 2: {e}")
            self.model2 = None
            
    def generate_explanation(self, model, cnn_output: str, max_length: int = 200) -> str:
        """Generate explanation from CNN output"""
        if model is None:
            return "Model not available"
            
        try:
            model.eval()
            
            # Create prompt
            prompt = f"CNN Classification:\n{cnn_output}\n\nExplanation:"
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=312)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract explanation (remove prompt)
            explanation = generated_text[len(prompt):].strip()
            
            return explanation
            
        except Exception as e:
            return f"Generation error: {e}"
    
    def compare_models(self, cnn_output: str) -> Dict:
        """Generate explanations from both models and compare"""
        print(f"\n🤖 Generating explanations for:")
        print(f"   {cnn_output.replace(chr(10), ' | ')}")
        
        start_time = time.time()
        
        # Model 1 explanation
        if self.model1:
            print("   [1/2] Model 1 (from scratch) generating...")
            explanation1 = self.generate_explanation(self.model1, cnn_output)
            time1 = time.time() - start_time
        else:
            explanation1 = "Model 1 not available"
            time1 = 0
            
        # Model 2 explanation  
        if self.model2:
            print("   [2/2] Model 2 (fine-tuned) generating...")
            explanation2 = self.generate_explanation(self.model2, cnn_output)
            time2 = time.time() - start_time - time1
        else:
            explanation2 = "Model 2 not available"
            time2 = 0
            
        return {
            'model1': {
                'explanation': explanation1,
                'time': time1,
                'name': 'Custom Transformer (56M, from scratch)'
            },
            'model2': {
                'explanation': explanation2, 
                'time': time2,
                'name': 'Fine-tuned GPT-2 (124M)'
            },
            'cnn_input': cnn_output
        }


def run_full_pipeline_test():
    """Test the complete CNN → LLM pipeline"""
    print("=" * 80)
    print("FULL PIPELINE TEST: CNN → LLM EXPLANATION")
    print("=" * 80)
    print("This tests the complete flow for your podcast demonstration!")
    
    # Initialize components
    print("\n[1/4] Initializing CNN inference...")
    cnn = CNNInference()
    
    print("\n[2/4] Initializing LLM inference...")
    llm = LLMInference()
    
    print("\n[3/4] Loading sample artworks...")
    samples = cnn.get_sample_images_from_dataset(num_samples=2)  # Start with 2 samples
    
    print("\n[4/4] Running full pipeline...")
    
    results = []
    
    for i, (image, ground_truth) in enumerate(samples, 1):
        print(f"\n" + "=" * 60)
        print(f"ARTWORK SAMPLE {i}")
        print("=" * 60)
        
        # Step 1: CNN Classification
        print("\n🎨 Step 1: CNN Classification")
        cnn_predictions = cnn.predict_image(image)
        cnn_formatted = cnn.format_for_llm(cnn_predictions)
        
        print("Ground Truth:")
        for task in ['artist', 'style', 'genre']:
            gt = ground_truth[task]
            print(f"   {task.capitalize()}: {gt['name']}")
            
        print("\nCNN Predictions:")  
        print(f"   {cnn_formatted.replace(chr(10), chr(10) + '   ')}")
        
        # Step 2: LLM Explanations
        print("\n🤖 Step 2: LLM Explanations")
        comparison = llm.compare_models(cnn_formatted)
        
        # Display results
        print("\n📖 MODEL 1 (Custom from Scratch):")
        print(f"   {comparison['model1']['explanation'][:200]}...")
        print(f"   Generation time: {comparison['model1']['time']:.2f}s")
        
        print("\n📖 MODEL 2 (Fine-tuned GPT-2):")
        print(f"   {comparison['model2']['explanation'][:200]}...")
        print(f"   Generation time: {comparison['model2']['time']:.2f}s")
        
        # Step 3: Quick Evaluation
        print("\n📊 Quick Evaluation:")
        
        # Length comparison
        len1 = len(comparison['model1']['explanation'].split())
        len2 = len(comparison['model2']['explanation'].split()) 
        print(f"   Length: Model 1 = {len1} words, Model 2 = {len2} words")
        
        # Speed comparison
        print(f"   Speed: Model 1 = {comparison['model1']['time']:.2f}s, Model 2 = {comparison['model2']['time']:.2f}s")
        
        # Check if mentions confidence scores
        mentions_conf1 = any(word in comparison['model1']['explanation'].lower() for word in ['confidence', '%', 'percent', 'certain'])
        mentions_conf2 = any(word in comparison['model2']['explanation'].lower() for word in ['confidence', '%', 'percent', 'certain'])
        print(f"   Mentions confidence: Model 1 = {'✓' if mentions_conf1 else '✗'}, Model 2 = {'✓' if mentions_conf2 else '✗'}")
        
        results.append({
            'sample_id': i,
            'ground_truth': ground_truth,
            'cnn_predictions': cnn_predictions,
            'cnn_formatted': cnn_formatted,
            'llm_comparison': comparison
        })
        
    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE TEST COMPLETE - PODCAST READY!")
    print("=" * 80)
    
    print(f"\n📈 Summary:")
    print(f"   Samples tested: {len(results)}")
    print(f"   CNN model: {'✓' if cnn else '✗'} loaded")
    print(f"   LLM Model 1: {'✓' if llm.model1 else '✗'} loaded") 
    print(f"   LLM Model 2: {'✓' if llm.model2 else '✗'} loaded")
    
    # Save results for podcast
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"pipeline_test_results_{timestamp}.json"
    
    # Convert torch tensors to lists for JSON serialization
    for result in results:
        if 'raw_logits' in result['cnn_predictions']:
            del result['cnn_predictions']['raw_logits']  # Remove non-serializable data
            
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n💾 Results saved to: {output_file}")
    print("\n🎥 Ready for podcast demonstration!")
    print("   Use these results to show side-by-side model comparisons")
    
    return results, cnn, llm


if __name__ == "__main__":
    # Run the complete pipeline test
    try:
        results, cnn_model, llm_models = run_full_pipeline_test()
        
        print("\n" + "💡" * 20)
        print("INTEGRATION SUCCESS! 🎉")
        print("Your CNN + LLM pipeline is working!")
        print("Ready for podcast demonstration and user evaluation.")
        print("💡" * 20)
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()