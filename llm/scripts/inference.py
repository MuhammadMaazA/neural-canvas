"""
Inference Script for Art Commentary
UCL COMP0220 Coursework - NST Art Commentary Project

Use trained LLMs to comment on NST-generated artwork
"""

import torch
import torch.nn.functional as F
import sys
import os
import re
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_model import GPT2FromScratch


class ArtCommentator:
    """Generate commentary for artwork using trained LLM"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.vocab = checkpoint['vocab']
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}
        config = checkpoint['config']
        
        # Initialize model
        self.model = GPT2FromScratch(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            max_len=config['max_len'],
            dropout=config.get('dropout', 0.1)
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
    
    def _tokenize(self, text: str):
        """Tokenize text"""
        text = text.lower().strip()
        words = re.findall(r'\b\w+\b', text)
        tokens = [self.vocab.get(word, 3) for word in words]
        return tokens
    
    def _detokenize(self, tokens):
        """Convert tokens to text"""
        words = []
        for idx in tokens:
            word = self.idx_to_word.get(idx, '<UNK>')
            if word not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                words.append(word)
        return ' '.join(words)
    
    def generate_commentary(self, prompt: str, max_length: int = 80, temperature: float = 0.8):
        """
        Generate art commentary from prompt
        
        Args:
            prompt: Input text (e.g., "This painting shows", "The style is")
            max_length: Maximum generation length
            temperature: Sampling temperature
        
        Returns:
            Generated commentary text
        """
        tokens = self._tokenize(prompt)
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(max_length):
                logits, _ = self.model(input_tensor, None)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Top-p sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > 0.9
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                if next_token in [0, 2]:  # PAD or EOS
                    break
                
                generated_tokens.append(next_token)
                input_tensor = torch.cat([
                    input_tensor,
                    torch.tensor([[next_token]], dtype=torch.long).to(self.device)
                ], dim=1)
                
                if input_tensor.size(1) > 256:
                    input_tensor = input_tensor[:, -256:]
        
        response = self._detokenize(generated_tokens)
        
        # Clean up
        if response:
            response = response[0].upper() + response[1:]
            if not response[-1] in ['.', '!', '?']:
                response += '.'
        
        return response
    
    def comment_on_nst(self, content_desc: str, style_desc: str):
        """
        Generate commentary for NST result
        
        Args:
            content_desc: Description of content image
            style_desc: Description of style image
        
        Returns:
            Commentary on the NST result
        """
        prompts = [
            f"The neural style transfer combines {content_desc} with {style_desc}",
            f"This artwork blends {content_desc} using the style of {style_desc}",
            f"By applying {style_desc} to {content_desc} the result",
        ]
        
        comments = []
        for prompt in prompts:
            comment = self.generate_commentary(prompt, max_length=60, temperature=0.7)
            comments.append(comment)
        
        return comments


def main():
    parser = argparse.ArgumentParser(description='Art Commentary Generator')
    parser.add_argument('--model', type=str, default='checkpoints/model1_best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'nst', 'demo'],
                       help='Mode of operation')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Art Commentary Generator")
    print("=" * 80)
    
    commentator = ArtCommentator(args.model)
    
    if args.mode == 'interactive':
        print("\nInteractive Mode (type 'quit' to exit)")
        print("Enter a prompt and I'll generate commentary\n")
        
        while True:
            prompt = input("Prompt: ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt.strip():
                continue
            
            commentary = commentator.generate_commentary(prompt)
            print(f"Commentary: {commentary}\n")
    
    elif args.mode == 'nst':
        print("\nNST Commentary Mode")
        content = input("Content image description: ")
        style = input("Style image description: ")
        
        print("\nGenerating commentary...")
        comments = commentator.comment_on_nst(content, style)
        
        print("\nGenerated Commentary:")
        for i, comment in enumerate(comments, 1):
            print(f"{i}. {comment}")
    
    elif args.mode == 'demo':
        print("\nDemo Mode - Example Commentaries\n")
        
        examples = [
            "This painting shows",
            "The artist used",
            "The style can be described as",
            "Looking at this artwork we can see",
            "The composition features",
        ]
        
        for example in examples:
            commentary = commentator.generate_commentary(example)
            print(f"Prompt: {example}")
            print(f"Commentary: {commentary}\n")


if __name__ == "__main__":
    main()
