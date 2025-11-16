#!/usr/bin/env python3
"""
Interactive Chat with Your Custom Art Expert Model
===================================================
Have a conversation with your trained model!
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from models.art_expert_model import create_art_expert_model


class ModelChat:
    def __init__(self, checkpoint_path):
        """Initialize the chat with model from checkpoint"""
        print("🤖 Loading Art Expert Model...")
        print(f"📁 Checkpoint: {checkpoint_path}\n")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        epoch = checkpoint.get('epoch', 'Unknown')
        val_loss = checkpoint.get('val_loss', 'Unknown')
        
        print(f"✓ Checkpoint loaded (Epoch: {epoch}, Val Loss: {val_loss})")
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✓ Tokenizer loaded")
        
        # Setup model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vocab_size = len(self.tokenizer)
        self.model = create_art_expert_model(vocab_size=vocab_size, model_size="base")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
        print()
        
    def generate(self, prompt, max_tokens=100, temperature=0.8, top_k=50, top_p=0.9):
        """Generate response to a prompt"""
        # Encode input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300)
        input_ids = inputs['input_ids'].to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def chat(self):
        """Interactive chat loop"""
        print("=" * 80)
        print("🎨 ART EXPERT CHATBOT - Interactive Mode")
        print("=" * 80)
        print()
        print("💬 Chat with your custom-trained art expert model!")
        print()
        print("📝 Commands:")
        print("  • Just type your question naturally")
        print("  • 'temp X' - Change temperature (0.1-2.0, default: 0.8)")
        print("  • 'tokens X' - Change max tokens (10-200, default: 100)")
        print("  • 'help' - Show example questions")
        print("  • 'quit' or 'exit' - End chat")
        print()
        print("=" * 80)
        print()
        
        # Settings
        temperature = 0.8
        max_tokens = 100
        
        # Show some examples
        self._show_examples()
        
        while True:
            try:
                # Get user input
                user_input = input("\n🧑 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                    print("\n👋 Thanks for chatting! Goodbye!\n")
                    break
                
                if user_input.lower() == 'help':
                    self._show_examples()
                    continue
                
                # Temperature command
                if user_input.lower().startswith('temp '):
                    try:
                        temp = float(user_input.split()[1])
                        if 0.1 <= temp <= 2.0:
                            temperature = temp
                            print(f"✓ Temperature set to {temperature}")
                        else:
                            print("❌ Temperature must be between 0.1 and 2.0")
                    except:
                        print("❌ Usage: temp 0.8")
                    continue
                
                # Max tokens command
                if user_input.lower().startswith('tokens '):
                    try:
                        tokens = int(user_input.split()[1])
                        if 10 <= tokens <= 200:
                            max_tokens = tokens
                            print(f"✓ Max tokens set to {max_tokens}")
                        else:
                            print("❌ Tokens must be between 10 and 200")
                    except:
                        print("❌ Usage: tokens 100")
                    continue
                
                # Format the prompt as Q&A
                if not user_input.startswith("Q:"):
                    prompt = f"Q: {user_input}\nA:"
                else:
                    if "\nA:" not in user_input:
                        prompt = f"{user_input}\nA:"
                    else:
                        prompt = user_input
                
                # Generate response
                print("\n🤖 Model: ", end="", flush=True)
                response = self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
                
                # Extract and clean the answer
                if "A:" in response:
                    # Get everything after the first A:
                    answer = response.split("A:", 1)[1].strip()
                    
                    # Stop at the next Q: if it appears
                    if "\nQ:" in answer:
                        answer = answer.split("\nQ:")[0].strip()
                    
                    # Stop at the next Explanation if it appears
                    if "\nExplanation:" in answer:
                        answer = answer.split("\nExplanation:")[0].strip()
                    
                    print(answer)
                else:
                    print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again with a different question.")
    
    def _show_examples(self):
        """Show example questions"""
        print("\n💡 Example questions you can ask:")
        print("  • What is Impressionism?")
        print("  • Tell me about Renaissance art")
        print("  • Who was Vincent van Gogh?")
        print("  • What makes a painting beautiful?")
        print("  • Explain abstract expressionism")
        print("  • What is the difference between modern and contemporary art?")
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Chat with your custom art expert model')
    parser.add_argument('--checkpoint', type=str, 
                       default='/cs/student/projects1/2023/muhamaaz/checkpoints/model1_custom/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--temp', type=float, default=0.8,
                       help='Temperature for generation (default: 0.8)')
    parser.add_argument('--tokens', type=int, default=100,
                       help='Max tokens to generate (default: 100)')
    
    args = parser.parse_args()
    
    # Create chat instance
    chat = ModelChat(args.checkpoint)
    
    # Start chatting
    chat.chat()


if __name__ == "__main__":
    main()
