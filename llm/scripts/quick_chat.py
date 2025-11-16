#!/usr/bin/env python3
"""
Quick Chat with Model 1 (Custom Art Expert)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from models.art_expert_model import create_art_expert_model

def load_model(checkpoint_path):
    """Load model from checkpoint"""
    print("Loading model...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create and load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = len(tokenizer)
    model = create_art_expert_model(vocab_size=vocab_size, model_size="base")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    return model, tokenizer, device


def generate_response(model, tokenizer, device, prompt, max_tokens=150, temperature=0.8):
    """Generate response to prompt"""
    # Encode
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
    input_ids = inputs['input_ids'].to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.9
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    checkpoint_path = '/cs/student/projects1/2023/muhamaaz/checkpoints/model1_custom/best_model.pt'
    
    print("=" * 80)
    print("ART EXPERT CHATBOT - Model 1 (Custom)")
    print("=" * 80)
    
    model, tokenizer, device = load_model(checkpoint_path)
    
    print("\n" + "=" * 80)
    print("CHAT INTERFACE")
    print("=" * 80)
    print("Type 'quit' or 'exit' to end the chat")
    print("Type 'temp X' to change temperature (e.g., 'temp 0.7')")
    print("-" * 80)
    
    temperature = 0.8
    
    # Example prompts to get started
    print("\n💡 Try these example prompts:")
    print("  - Q: What is Impressionism?")
    print("  - Q: Tell me about Renaissance art.")
    print("  - Q: What is abstract art?")
    print("  - Q: Who was Vincent van Gogh?")
    print("-" * 80 + "\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            # Check for exit
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            # Check for temperature change
            if user_input.lower().startswith('temp '):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"✓ Temperature set to {temperature}")
                    continue
                except:
                    print("❌ Invalid temperature. Use: temp 0.7")
                    continue
            
            # Format as Q&A if not already
            if not user_input.startswith("Q:"):
                prompt = f"Q: {user_input}\nA:"
            else:
                if "\nA:" not in user_input:
                    prompt = f"{user_input}\nA:"
                else:
                    prompt = user_input
            
            # Generate response
            print("\nModel: ", end="", flush=True)
            response = generate_response(model, tokenizer, device, prompt, 
                                        max_tokens=100, temperature=temperature)
            
            # Extract just the answer part
            if "A:" in response:
                answer = response.split("A:", 1)[1].strip()
                # Clean up - stop at next Q: if present
                if "\nQ:" in answer:
                    answer = answer.split("\nQ:")[0].strip()
                print(answer)
            else:
                print(response)
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()
