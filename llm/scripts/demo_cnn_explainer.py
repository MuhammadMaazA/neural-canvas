#!/usr/bin/env python3
"""
CNN Explainer Demo - Compare Model 1 vs Model 2
=================================================
Interactive demo comparing:
- Model 1: Custom transformer trained from scratch
- Model 2: Fine-tuned DistilGPT-2

Takes CNN classification outputs and generates natural language explanations.
Perfect for coursework demonstration and podcast recording.
"""

import os
os.environ['HF_HOME'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['HF_DATASETS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'

import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.art_expert_model import create_art_expert_model


class CNNExplainerDemo:
    """Demo class for comparing both CNN explainer models"""
    
    def __init__(
        self,
        model1_path: str = "/cs/student/projects1/2023/muhamaaz/checkpoints/model1_cnn_explainer/best_model.pt",
        model2_path: str = "/cs/student/projects1/2023/muhamaaz/checkpoints/model2_cnn_explainer/best_model_hf",
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model1 = None
        self.model2 = None
        self.tokenizer = None
        
        # Load tokenizer (same for both)
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load Model 1 (from scratch)
        if os.path.exists(model1_path):
            print(f"\nLoading Model 1 (from scratch): {model1_path}")
            try:
                checkpoint = torch.load(model1_path, map_location=self.device, weights_only=False)
                self.model1 = create_art_expert_model(
                    self.tokenizer.vocab_size,
                    checkpoint['config'].get('model_size', 'base')
                ).to(self.device)
                self.model1.load_state_dict(checkpoint['model_state_dict'])
                self.model1.eval()
                params = sum(p.numel() for p in self.model1.parameters())
                print(f"✓ Model 1 loaded ({params/1e6:.1f}M params)")
            except Exception as e:
                print(f"✗ Could not load Model 1: {e}")
        else:
            print(f"⚠ Model 1 not found at: {model1_path}")
        
        # Load Model 2 (fine-tuned)
        if os.path.exists(model2_path):
            print(f"\nLoading Model 2 (fine-tuned): {model2_path}")
            try:
                self.model2 = AutoModelForCausalLM.from_pretrained(model2_path)
                self.model2 = self.model2.to(self.device)
                self.model2.eval()
                params = sum(p.numel() for p in self.model2.parameters())
                print(f"✓ Model 2 loaded ({params/1e6:.1f}M params)")
            except Exception as e:
                print(f"✗ Could not load Model 2: {e}")
        else:
            print(f"⚠ Model 2 not found at: {model2_path}")
        
        print("\n" + "=" * 60)
        print("READY FOR DEMONSTRATION")
        print("=" * 60)
    
    def format_cnn_output(self, artist: str, style: str, genre: str,
                          artist_conf: float, style_conf: float, genre_conf: float) -> str:
        """Format CNN output as prompt for LLM"""
        return f"""The CNN classified this artwork as:
- Artist: {artist} ({artist_conf:.1%} confidence)
- Style: {style} ({style_conf:.1%} confidence)
- Genre: {genre} ({genre_conf:.1%} confidence)

Explain this classification:"""
    
    @torch.no_grad()
    def generate_model1(self, prompt: str, max_new_tokens: int = 150,
                        temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9) -> str:
        """Generate explanation using Model 1 (from scratch)"""
        if self.model1 is None:
            return "[Model 1 not loaded]"
        
        # Tokenize
        tokens = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device)
        
        # Generate
        generated = self.model1.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # Decode
        output = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Extract the generated part (after the prompt)
        if prompt in output:
            output = output[len(prompt):].strip()
        
        return output
    
    @torch.no_grad()
    def generate_model2(self, prompt: str, max_new_tokens: int = 150,
                        temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9) -> str:
        """Generate explanation using Model 2 (fine-tuned)"""
        if self.model2 is None:
            return "[Model 2 not loaded]"
        
        # Tokenize
        tokens = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device)
        
        # Generate
        generated = self.model2.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode
        output = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Extract the generated part (after the prompt)
        if prompt in output:
            output = output[len(prompt):].strip()
        
        return output
    
    def explain_cnn_output(self, artist: str, style: str, genre: str,
                           artist_conf: float = 0.85, style_conf: float = 0.90, genre_conf: float = 0.80,
                           max_new_tokens: int = 150) -> dict:
        """
        Generate explanations from both models for a CNN output
        
        Args:
            artist: Predicted artist name
            style: Predicted art style
            genre: Predicted genre
            artist_conf: Artist prediction confidence (0-1)
            style_conf: Style prediction confidence (0-1)
            genre_conf: Genre prediction confidence (0-1)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with explanations from both models
        """
        # Format the prompt
        prompt = self.format_cnn_output(artist, style, genre, artist_conf, style_conf, genre_conf)
        
        # Generate from both models
        print("\nGenerating explanations...")
        
        explanation1 = self.generate_model1(prompt, max_new_tokens)
        explanation2 = self.generate_model2(prompt, max_new_tokens)
        
        return {
            'prompt': prompt,
            'model1_explanation': explanation1,
            'model2_explanation': explanation2
        }
    
    def demo_comparison(self):
        """Run interactive comparison demo"""
        print("\n" + "=" * 80)
        print("CNN EXPLAINER DEMO - MODEL COMPARISON")
        print("=" * 80)
        print("This demo compares explanations from:")
        print("  Model 1: Custom transformer (trained from scratch)")
        print("  Model 2: DistilGPT-2 (fine-tuned)")
        print("=" * 80)
        
        # Sample CNN outputs to demonstrate
        samples = [
            {
                "artist": "Vincent van Gogh",
                "style": "Post-Impressionism",
                "genre": "Landscape",
                "artist_conf": 0.87,
                "style_conf": 0.92,
                "genre_conf": 0.78
            },
            {
                "artist": "Claude Monet",
                "style": "Impressionism",
                "genre": "Landscape",
                "artist_conf": 0.91,
                "style_conf": 0.95,
                "genre_conf": 0.85
            },
            {
                "artist": "Pablo Picasso",
                "style": "Cubism",
                "genre": "Portrait",
                "artist_conf": 0.76,
                "style_conf": 0.88,
                "genre_conf": 0.72
            },
            {
                "artist": "Gustav Klimt",
                "style": "Art Nouveau",
                "genre": "Portrait",
                "artist_conf": 0.83,
                "style_conf": 0.79,
                "genre_conf": 0.91
            },
            {
                "artist": "Salvador Dalí",
                "style": "Surrealism",
                "genre": "Abstract",
                "artist_conf": 0.89,
                "style_conf": 0.93,
                "genre_conf": 0.67
            }
        ]
        
        for i, sample in enumerate(samples, 1):
            print(f"\n{'='*80}")
            print(f"SAMPLE {i}/{len(samples)}")
            print(f"{'='*80}")
            
            results = self.explain_cnn_output(**sample)
            
            print("\n📊 CNN OUTPUT:")
            print(f"   Artist: {sample['artist']} ({sample['artist_conf']:.1%})")
            print(f"   Style:  {sample['style']} ({sample['style_conf']:.1%})")
            print(f"   Genre:  {sample['genre']} ({sample['genre_conf']:.1%})")
            
            print("\n🤖 MODEL 1 (From Scratch):")
            print("-" * 40)
            print(results['model1_explanation'][:500])
            
            print("\n🎯 MODEL 2 (Fine-tuned):")
            print("-" * 40)
            print(results['model2_explanation'][:500])
            
            if i < len(samples):
                input("\nPress Enter for next sample...")
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETE")
        print("=" * 80)
    
    def interactive_mode(self):
        """Run interactive mode where user can input CNN predictions"""
        print("\n" + "=" * 80)
        print("INTERACTIVE MODE")
        print("=" * 80)
        print("Enter CNN predictions to get explanations from both models.")
        print("Type 'quit' to exit.")
        print("=" * 80)
        
        while True:
            print("\nEnter CNN predictions:")
            
            artist = input("  Artist: ").strip()
            if artist.lower() == 'quit':
                break
            
            style = input("  Style: ").strip()
            if style.lower() == 'quit':
                break
            
            genre = input("  Genre: ").strip()
            if genre.lower() == 'quit':
                break
            
            try:
                artist_conf = float(input("  Artist confidence (0-1): "))
                style_conf = float(input("  Style confidence (0-1): "))
                genre_conf = float(input("  Genre confidence (0-1): "))
            except ValueError:
                artist_conf, style_conf, genre_conf = 0.85, 0.90, 0.80
                print("  Using default confidence values")
            
            results = self.explain_cnn_output(artist, style, genre, 
                                              artist_conf, style_conf, genre_conf)
            
            print("\n🤖 MODEL 1 (From Scratch):")
            print("-" * 40)
            print(results['model1_explanation'][:600])
            
            print("\n🎯 MODEL 2 (Fine-tuned):")
            print("-" * 40)
            print(results['model2_explanation'][:600])
        
        print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="CNN Explainer Demo")
    parser.add_argument("--mode", choices=["demo", "interactive"], default="demo",
                       help="Mode: 'demo' for sample comparisons, 'interactive' for custom input")
    parser.add_argument("--model1", type=str, 
                       default="/cs/student/projects1/2023/muhamaaz/checkpoints/model1_cnn_explainer/best_model.pt",
                       help="Path to Model 1 checkpoint")
    parser.add_argument("--model2", type=str,
                       default="/cs/student/projects1/2023/muhamaaz/checkpoints/model2_cnn_explainer/best_model_hf",
                       help="Path to Model 2 checkpoint")
    args = parser.parse_args()
    
    # Initialize demo
    demo = CNNExplainerDemo(
        model1_path=args.model1,
        model2_path=args.model2
    )
    
    # Run selected mode
    if args.mode == "demo":
        demo.demo_comparison()
    else:
        demo.interactive_mode()


if __name__ == "__main__":
    main()

