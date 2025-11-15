"""
Interactive Chat Interface for Art Expert Models
================================================
Chat with Model 1 (custom) and Model 2 (pretrained) to compare responses

Usage:
    python chat_with_models.py --model model1
    python chat_with_models.py --model model2
    python chat_with_models.py --model both  (compare side-by-side)
"""

import os
os.environ['HF_HOME'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['HF_DATASETS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.art_expert_model import create_art_expert_model


class ArtExpertChatbot:
    """Chatbot interface for trained models"""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def generate_response(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> str:
        """Generate response to prompt"""

        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)

        # Generate
        output_ids = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Decode
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Remove prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response


def load_model1(checkpoint_path: str, device: str = "cuda"):
    """Load custom trained model (Model 1)"""
    print(f"\nLoading Model 1 (Custom) from {checkpoint_path}...")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model 1 checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load tokenizer
    tokenizer_name = checkpoint['config'].get('tokenizer_name', 'gpt2')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create model
    model_size = checkpoint['config'].get('model_size', 'base')
    model = create_art_expert_model(tokenizer.vocab_size, model_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model 1 loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Val Loss: {checkpoint.get('loss', 'N/A'):.4f}" if isinstance(checkpoint.get('loss'), float) else "")

    return model, tokenizer


def load_model2(checkpoint_path: str, device: str = "cuda"):
    """Load fine-tuned pretrained model (Model 2)"""
    print(f"\nLoading Model 2 (Pretrained) from {checkpoint_path}...")

    # Try HuggingFace format first
    hf_path = checkpoint_path.replace('.pt', '_hf')
    if os.path.exists(hf_path):
        print(f"Loading from HuggingFace format: {hf_path}")
        model = AutoModelForCausalLM.from_pretrained(hf_path)
        tokenizer = AutoTokenizer.from_pretrained(hf_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = model.to(device)
        model.eval()
        print(f"✓ Model 2 loaded successfully (HF format)")
    else:
        # Load from checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model 2 checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model and tokenizer
        model_name = checkpoint['config'].get('model_name', 'distilgpt2')
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        print(f"✓ Model 2 loaded successfully")
        print(f"  Base: {model_name}")
        print(f"  Epoch: {checkpoint['epoch'] + 1}")
        print(f"  Val Loss: {checkpoint.get('loss', 'N/A'):.4f}" if isinstance(checkpoint.get('loss'), float) else "")

    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    return model, tokenizer


def chat_single_model(chatbot: ArtExpertChatbot, model_name: str):
    """Interactive chat with a single model"""
    print("\n" + "=" * 80)
    print(f"CHAT WITH {model_name.upper()}")
    print("=" * 80)
    print("Type 'quit' or 'exit' to stop")
    print("Type 'clear' to reset conversation")
    print("=" * 80 + "\n")

    conversation_history = ""

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if user_input.lower() == 'clear':
                conversation_history = ""
                print("\n[Conversation cleared]")
                continue

            if not user_input:
                continue

            # Build prompt with conversation history
            if conversation_history:
                prompt = f"{conversation_history}\nUser: {user_input}\nAssistant:"
            else:
                prompt = f"User: {user_input}\nAssistant:"

            # Generate response
            print("\n" + model_name + ":", end=" ", flush=True)
            response = chatbot.generate_response(
                prompt,
                max_length=150,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )

            print(response)

            # Update conversation history (keep last 3 exchanges)
            conversation_history += f"\nUser: {user_input}\nAssistant: {response}"
            # Keep only last 500 tokens to avoid context overflow
            if len(conversation_history) > 2000:
                conversation_history = conversation_history[-2000:]

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


def chat_compare_models(chatbot1: ArtExpertChatbot, chatbot2: ArtExpertChatbot):
    """Compare responses from both models side-by-side"""
    print("\n" + "=" * 80)
    print("COMPARE MODELS - SIDE BY SIDE")
    print("=" * 80)
    print("Type 'quit' or 'exit' to stop")
    print("=" * 80 + "\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            prompt = f"User: {user_input}\nAssistant:"

            # Generate from Model 1
            print("\n" + "-" * 80)
            print("MODEL 1 (Custom):")
            print("-" * 80)
            response1 = chatbot1.generate_response(
                prompt,
                max_length=150,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            print(response1)

            # Generate from Model 2
            print("\n" + "-" * 80)
            print("MODEL 2 (Pretrained):")
            print("-" * 80)
            response2 = chatbot2.generate_response(
                prompt,
                max_length=150,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
            print(response2)
            print("-" * 80)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


def test_predefined_questions(chatbot: ArtExpertChatbot, model_name: str):
    """Test model with predefined art and AI literacy questions"""
    questions = [
        # Art questions
        "What is Impressionism?",
        "Tell me about Claude Monet.",
        "What defines Cubism art?",
        "Explain the difference between Renaissance and Baroque art.",

        # AI literacy questions
        "What is a neural network?",
        "How does AI learn?",
        "Explain like I'm five: what is machine learning?",
        "Can computers really understand art?",

        # Integration questions
        "How can AI recognize art styles?",
        "What is a CNN and how does it see images?",
        "Why is deep learning called deep?",
    ]

    print("\n" + "=" * 80)
    print(f"TESTING {model_name.upper()} - PREDEFINED QUESTIONS")
    print("=" * 80 + "\n")

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Q: {question}")
        print("-" * 80)

        prompt = f"Question: {question}\nAnswer:"
        response = chatbot.generate_response(
            prompt,
            max_length=150,
            temperature=0.7,
            top_k=40,
            top_p=0.9
        )

        print(f"A: {response}")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Chat with Art Expert Models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["model1", "model2", "both"],
        default="both",
        help="Which model to use (model1=custom, model2=pretrained, both=compare)"
    )
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
        "--test",
        action="store_true",
        help="Run predefined test questions instead of interactive chat"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("NEURAL CANVAS - ART EXPERT CHATBOT")
    print("=" * 80)
    print(f"Device: {args.device}")
    print("=" * 80)

    # Load models
    chatbot1 = None
    chatbot2 = None

    if args.model in ["model1", "both"]:
        try:
            model1, tokenizer1 = load_model1(args.model1_path, args.device)
            chatbot1 = ArtExpertChatbot(model1, tokenizer1, args.device)
        except Exception as e:
            print(f"Error loading Model 1: {e}")
            if args.model == "model1":
                return

    if args.model in ["model2", "both"]:
        try:
            model2, tokenizer2 = load_model2(args.model2_path, args.device)
            chatbot2 = ArtExpertChatbot(model2, tokenizer2, args.device)
        except Exception as e:
            print(f"Error loading Model 2: {e}")
            if args.model == "model2":
                return

    # Run chat or tests
    if args.test:
        if chatbot1:
            test_predefined_questions(chatbot1, "Model 1 (Custom)")
        if chatbot2:
            test_predefined_questions(chatbot2, "Model 2 (Pretrained)")
    else:
        if args.model == "model1" and chatbot1:
            chat_single_model(chatbot1, "Model 1 (Custom)")
        elif args.model == "model2" and chatbot2:
            chat_single_model(chatbot2, "Model 2 (Pretrained)")
        elif args.model == "both" and chatbot1 and chatbot2:
            chat_compare_models(chatbot1, chatbot2)


if __name__ == "__main__":
    main()
