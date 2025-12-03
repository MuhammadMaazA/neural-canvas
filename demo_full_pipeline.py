#!/usr/bin/env python3
"""
FULL PIPELINE DEMO: CNN → LLM Explanation
==========================================
Complete demonstration pipeline:
1. CNN classifies artwork (artist, style, genre)
2. LLM explains the classification

This is for podcast demonstration and coursework presentation.

Usage:
    python demo_full_pipeline.py                    # Demo with WikiArt samples
    python demo_full_pipeline.py --image path.jpg  # Demo with custom image
"""

import os
os.environ['HF_HOME'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['HF_DATASETS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'

import sys
import torch
import argparse
from PIL import Image
import numpy as np

# Add paths
sys.path.append('/cs/student/projects1/2023/muhamaaz/neural-canvas')
sys.path.append('/cs/student/projects1/2023/muhamaaz/neural-canvas/llm')

from cnn_inference import CNNInference
from llm.scripts.demo_cnn_explainer import CNNExplainerDemo


class FullPipelineDemo:
    """
    Full demonstration pipeline combining CNN + LLM
    
    Flow:
    1. Image → CNN → (artist, style, genre) predictions
    2. Predictions → LLM → Natural language explanation
    """
    
    def __init__(
        self,
        cnn_checkpoint: str = None,
        model1_path: str = "/cs/student/projects1/2023/muhamaaz/checkpoints/model1_cnn_explainer/best_model.pt",
        model2_path: str = "/cs/student/projects1/2023/muhamaaz/checkpoints/model2_cnn_explainer/best_model_hf"
    ):
        print("=" * 80)
        print("NEURAL CANVAS - FULL PIPELINE DEMO")
        print("=" * 80)
        print("CNN Image Classification → LLM Explanation Generation")
        print("=" * 80)
        
        # Initialize CNN
        print("\n[1/2] Loading CNN model...")
        self.cnn = CNNInference(checkpoint_path=cnn_checkpoint)
        
        # Initialize LLM explainer
        print("\n[2/2] Loading LLM models...")
        self.llm = CNNExplainerDemo(
            model1_path=model1_path,
            model2_path=model2_path
        )
        
        print("\n" + "=" * 80)
        print("PIPELINE READY")
        print("=" * 80)
    
    def process_image(self, image, ground_truth: dict = None, max_tokens: int = 150):
        """
        Process a single image through the full pipeline
        
        Args:
            image: PIL Image or path to image
            ground_truth: Optional dict with 'artist', 'style', 'genre' ground truth
            max_tokens: Max tokens for LLM generation
            
        Returns:
            Dictionary with CNN predictions and LLM explanations
        """
        print("\n" + "-" * 60)
        print("PROCESSING IMAGE")
        print("-" * 60)
        
        # Step 1: CNN Classification
        print("\n🔍 Step 1: CNN Classification")
        predictions = self.cnn.predict_image(image)
        
        artist = predictions['artist']['name']
        style = predictions['style']['name']
        genre = predictions['genre']['name']
        artist_conf = predictions['artist']['confidence']
        style_conf = predictions['style']['confidence']
        genre_conf = predictions['genre']['confidence']
        
        print(f"   Artist: {artist} ({artist_conf:.1%})")
        print(f"   Style:  {style} ({style_conf:.1%})")
        print(f"   Genre:  {genre} ({genre_conf:.1%})")
        
        # Show ground truth if available
        if ground_truth:
            print("\n📋 Ground Truth:")
            print(f"   Artist: {ground_truth.get('artist', 'N/A')}")
            print(f"   Style:  {ground_truth.get('style', 'N/A')}")
            print(f"   Genre:  {ground_truth.get('genre', 'N/A')}")
            
            # Check accuracy
            print("\n✅ Accuracy:")
            for task in ['artist', 'style', 'genre']:
                pred = predictions[task]['name']
                gt = ground_truth.get(task, '')
                correct = pred.lower() == gt.lower() if gt else False
                print(f"   {task.capitalize()}: {'✓' if correct else '✗'}")
        
        # Step 2: LLM Explanation
        print("\n💬 Step 2: LLM Explanation Generation")
        
        explanations = self.llm.explain_cnn_output(
            artist=artist,
            style=style,
            genre=genre,
            artist_conf=artist_conf,
            style_conf=style_conf,
            genre_conf=genre_conf,
            max_new_tokens=max_tokens
        )
        
        print("\n🤖 Model 1 (From Scratch) says:")
        print("-" * 40)
        print(explanations['model1_explanation'][:500])
        
        print("\n🎯 Model 2 (Fine-tuned) says:")
        print("-" * 40)
        print(explanations['model2_explanation'][:500])
        
        return {
            'predictions': predictions,
            'explanations': explanations,
            'ground_truth': ground_truth
        }
    
    def demo_with_wikiart(self, num_samples: int = 3):
        """Run demo with samples from WikiArt dataset"""
        print("\n" + "=" * 80)
        print("DEMO: WikiArt Sample Images")
        print("=" * 80)
        
        # Get sample images from WikiArt
        samples = self.cnn.get_sample_images_from_dataset(num_samples=num_samples)
        
        results = []
        for i, (image, ground_truth) in enumerate(samples, 1):
            print(f"\n{'='*80}")
            print(f"SAMPLE {i}/{num_samples}")
            print(f"{'='*80}")
            
            # Convert ground truth to simple dict
            gt = {
                'artist': ground_truth['artist']['name'],
                'style': ground_truth['style']['name'],
                'genre': ground_truth['genre']['name']
            }
            
            result = self.process_image(image, ground_truth=gt)
            results.append(result)
            
            if i < num_samples:
                input("\nPress Enter for next sample...")
        
        return results
    
    def demo_with_custom_image(self, image_path: str):
        """Run demo with a custom image file"""
        print("\n" + "=" * 80)
        print("DEMO: Custom Image")
        print("=" * 80)
        print(f"Image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return None
        
        return self.process_image(image_path)


def main():
    parser = argparse.ArgumentParser(
        description="Full Pipeline Demo: CNN → LLM Explanation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo_full_pipeline.py                       # Demo with 3 WikiArt samples
    python demo_full_pipeline.py --samples 5           # Demo with 5 WikiArt samples
    python demo_full_pipeline.py --image artwork.jpg   # Demo with custom image
        """
    )
    parser.add_argument("--image", type=str, help="Path to custom image file")
    parser.add_argument("--samples", type=int, default=3, help="Number of WikiArt samples (default: 3)")
    parser.add_argument("--cnn-checkpoint", type=str, help="Path to CNN checkpoint")
    parser.add_argument("--model1", type=str,
                       default="/cs/student/projects1/2023/muhamaaz/checkpoints/model1_cnn_explainer/best_model.pt",
                       help="Path to Model 1 (from scratch)")
    parser.add_argument("--model2", type=str,
                       default="/cs/student/projects1/2023/muhamaaz/checkpoints/model2_cnn_explainer/best_model_hf",
                       help="Path to Model 2 (fine-tuned)")
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FullPipelineDemo(
        cnn_checkpoint=args.cnn_checkpoint,
        model1_path=args.model1,
        model2_path=args.model2
    )
    
    # Run appropriate demo
    if args.image:
        pipeline.demo_with_custom_image(args.image)
    else:
        pipeline.demo_with_wikiart(num_samples=args.samples)
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nFor your podcast, you can:")
    print("1. Show CNN classifying artworks")
    print("2. Compare explanations from Model 1 (scratch) vs Model 2 (fine-tuned)")
    print("3. Discuss differences in output quality and training approaches")
    print("=" * 80)


if __name__ == "__main__":
    main()

