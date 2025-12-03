#!/usr/bin/env python3
"""
CNN Inference Script for Integration with LLM Models
====================================================

This script loads a trained CNN model and provides inference capabilities
to generate predictions that can be fed into your LLM explanation models.

Integrates with your podcast demonstration pipeline:
CNN Image Classification → LLM Explanation Generation
"""

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import sys
import os
sys.path.append('/cs/student/projects1/2023/muhamaaz/neural-canvas/cnn_models')

from cnn_models.model import build_model
from cnn_models.config import Config
from datasets import load_dataset
import numpy as np
from typing import Dict, Tuple, List, Union
import json


class CNNInference:
    """Loads trained CNN and provides inference for LLM integration"""
    
    def __init__(self, checkpoint_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load config
        self.config = Config()
        
        # Load dataset to get class mappings
        print("Loading dataset for class mappings...")
        self.dataset = load_dataset(self.config.dataset_name, split="train", streaming=True)
        self.artist_names = self.dataset.features['artist'].names
        self.style_names = self.dataset.features['style'].names  
        self.genre_names = self.dataset.features['genre'].names
        
        print(f"Artists: {len(self.artist_names)} classes")
        print(f"Styles: {len(self.style_names)} classes")
        print(f"Genres: {len(self.genre_names)} classes")
        
        # Create model
        self.num_classes = {
            'artist': len(self.artist_names),
            'style': len(self.style_names), 
            'genre': len(self.genre_names)
        }
        
        self.model = build_model(self.config, self.num_classes)
        self.model.to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        else:
            # Try to find best checkpoint automatically
            checkpoint_dir = 'cnn_models/checkpoints'
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
                if checkpoints:
                    # Find checkpoint with highest macro score
                    best_checkpoint = max(checkpoints, key=lambda x: float(x.split('macro')[1].split('.pt')[0]))
                    checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint)
                    print(f"Auto-detected best checkpoint: {checkpoint_path}")
                    self.load_checkpoint(checkpoint_path)
                else:
                    print("No checkpoints found - using untrained model")
            else:
                print("No checkpoint directory found - using untrained model")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            print(f"✅ Loaded checkpoint: {checkpoint_path}")
            print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"   Macro Acc: {checkpoint.get('macro_acc', 'unknown'):.4f}")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            
    def predict_image(self, image_path_or_pil: Union[str, Image.Image]) -> Dict:
        """
        Predict artist, style, genre for a single image
        
        Returns formatted output suitable for LLM input:
        {
            'artist': {'name': 'Vincent van Gogh', 'confidence': 0.873, 'class_id': 42},
            'style': {'name': 'Post-Impressionism', 'confidence': 0.912, 'class_id': 15}, 
            'genre': {'name': 'Landscape', 'confidence': 0.835, 'class_id': 8},
            'raw_logits': {...}  # For debugging
        }
        """
        self.model.eval()
        
        # Load and preprocess image
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert('RGB')
        else:
            image = image_path_or_pil
            
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(image_tensor)
            
        # Convert to probabilities
        probs = {task: F.softmax(logits[task], dim=1) for task in logits}
        
        # Get top prediction for each task
        results = {}
        for task, prob in probs.items():
            top_prob, top_idx = torch.max(prob, dim=1)
            confidence = top_prob.item()
            class_id = top_idx.item()
            
            # Map to class name
            if task == 'artist':
                class_name = self.artist_names[class_id]
            elif task == 'style':
                class_name = self.style_names[class_id] 
            elif task == 'genre':
                class_name = self.genre_names[class_id]
                
            results[task] = {
                'name': class_name,
                'confidence': confidence,
                'class_id': class_id
            }
            
        # Add raw logits for debugging
        results['raw_logits'] = {task: logits[task].cpu().numpy().tolist() for task in logits}
        
        return results
    
    def format_for_llm(self, predictions: Dict) -> str:
        """
        Format CNN predictions into text suitable for LLM input
        
        Example output:
        "Artist: Vincent van Gogh (87.3% confidence)
         Style: Post-Impressionism (91.2% confidence)  
         Genre: Landscape (83.5% confidence)"
        """
        formatted_lines = []
        for task in ['artist', 'style', 'genre']:
            if task in predictions:
                pred = predictions[task]
                confidence_pct = pred['confidence'] * 100
                formatted_lines.append(f"{task.capitalize()}: {pred['name']} ({confidence_pct:.1f}% confidence)")
                
        return "\n".join(formatted_lines)
    
    def get_sample_images_from_dataset(self, num_samples: int = 5) -> List[Tuple[Image.Image, Dict]]:
        """Get sample images from WikiArt dataset for testing"""
        print(f"Loading {num_samples} sample images from WikiArt...")
        
        # Get the full dataset (not streaming for sampling)
        dataset = load_dataset(self.config.dataset_name, split="test")
        
        # Sample random indices
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        samples = []
        for idx in indices:
            item = dataset[int(idx)]
            image = item['image']
            
            # Ground truth for comparison
            ground_truth = {
                'artist': {'name': self.artist_names[item['artist']], 'class_id': item['artist']},
                'style': {'name': self.style_names[item['style']], 'class_id': item['style']},
                'genre': {'name': self.genre_names[item['genre']], 'class_id': item['genre']}
            }
            
            samples.append((image, ground_truth))
            
        return samples


def demo_cnn_llm_integration():
    """Demonstrate CNN + LLM integration pipeline"""
    print("=" * 80)
    print("CNN + LLM INTEGRATION DEMO")
    print("=" * 80)
    
    # Initialize CNN
    cnn = CNNInference()
    
    # Get sample images
    print("\n[1] Loading sample images from WikiArt...")
    samples = cnn.get_sample_images_from_dataset(num_samples=3)
    
    print("\n[2] Running CNN predictions...")
    for i, (image, ground_truth) in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        
        # CNN prediction
        predictions = cnn.predict_image(image)
        
        # Format for LLM
        llm_input = cnn.format_for_llm(predictions)
        
        # Display results
        print("🎯 Ground Truth:")
        for task in ['artist', 'style', 'genre']:
            gt = ground_truth[task]
            print(f"   {task.capitalize()}: {gt['name']}")
            
        print("\n🤖 CNN Predictions:")
        print(f"   {llm_input}")
        
        print("\n📝 LLM Input Format:")
        print(f'   """{llm_input}"""')
        
        # Check accuracy
        print("\n✅ Accuracy Check:")
        for task in ['artist', 'style', 'genre']:
            pred_name = predictions[task]['name']
            gt_name = ground_truth[task]['name'] 
            correct = pred_name == gt_name
            confidence = predictions[task]['confidence'] * 100
            print(f"   {task.capitalize()}: {'✓' if correct else '✗'} ({confidence:.1f}% confidence)")
    
    print("\n[3] Integration with LLM models...")
    print("💡 Next steps:")
    print("   - Feed CNN output to your trained LLM models")
    print("   - Compare Model 1 (from scratch) vs Model 2 (fine-tuned) explanations")
    print("   - Use this for podcast demonstration")
    
    return cnn


if __name__ == "__main__":
    # Run the demo
    cnn_model = demo_cnn_llm_integration()
    
    print("\n" + "=" * 80)
    print("CNN INFERENCE READY FOR LLM INTEGRATION")
    print("=" * 80)
    print("Use the CNNInference class in your LLM scripts:")
    print("```python")
    print("from cnn_inference import CNNInference")
    print("cnn = CNNInference()")
    print("predictions = cnn.predict_image('path/to/image.jpg')")
    print("llm_input = cnn.format_for_llm(predictions)")
    print("```")