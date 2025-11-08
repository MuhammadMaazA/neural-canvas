"""
Conversational AI Agent for AI Literacy Podcast
UCL COMP0220 Coursework - Conversational Agent Module

This module implements a conversational AI agent specifically designed
for the AI literacy podcast. It handles:
- Natural dialogue generation
- AI concept explanations
- Interactive Q&A
- Personality and tone management
- Conversation history tracking
"""

import torch
import torch.nn.functional as F
import re
from typing import List, Dict, Tuple, Optional
import json


class ConversationalAgent:
    """
    AI Agent for educational podcast conversations
    
    Features:
    - Maintains conversation context
    - Adapts explanations to audience level
    - Tracks dialogue history
    - Generates engaging responses
    """
    
    def __init__(self, model, vocab: Dict[str, int], device: str = 'cuda',
                 personality: str = 'friendly_educator', max_context_length: int = 512):
        """
        Initialize the conversational agent
        
        Args:
            model: Trained language model (GPT-2 style or fine-tuned)
            vocab: Vocabulary dictionary
            device: torch device
            personality: Agent personality ('friendly_educator', 'technical_expert', 'curious_learner')
            max_context_length: Maximum context window size
        """
        self.model = model
        self.vocab = vocab
        self.idx_to_word = {idx: word for word, idx in vocab.items()}
        self.device = device
        self.personality = personality
        self.max_context_length = max_context_length
        
        # Conversation history
        self.conversation_history = []
        self.context_window = []
        
        # Personality templates
        self.personality_prompts = {
            'friendly_educator': "As a friendly AI educator, I'll help you understand AI concepts in simple terms.",
            'technical_expert': "I'll provide detailed technical explanations of AI and machine learning concepts.",
            'curious_learner': "Let's explore AI together! I'm curious about these concepts too."
        }
        
        # Special tokens
        self.PAD_IDX = 0
        self.SOS_IDX = 1
        self.EOS_IDX = 2
        self.UNK_IDX = 3
        
        self.model.eval()
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        self.context_window = []
        print("💬 Conversation reset. Starting fresh!")
    
    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text to vocabulary indices"""
        text = text.lower().strip()
        words = re.findall(r'\b\w+\b', text)
        tokens = [self.vocab.get(word, self.UNK_IDX) for word in words]
        return tokens
    
    def _detokenize(self, tokens: List[int]) -> str:
        """Convert tokens back to text"""
        words = []
        for idx in tokens:
            word = self.idx_to_word.get(idx, '<UNK>')
            if word not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                words.append(word)
        return ' '.join(words)
    
    def _build_context(self, current_input: str) -> str:
        """Build context from conversation history"""
        # Include recent conversation turns
        context_parts = []
        
        # Add personality prompt
        if self.personality in self.personality_prompts:
            context_parts.append(self.personality_prompts[self.personality])
        
        # Add recent conversation history (last 3 turns)
        recent_history = self.conversation_history[-3:] if len(self.conversation_history) > 0 else []
        for turn in recent_history:
            context_parts.append(f"Human: {turn['human']}")
            context_parts.append(f"AI: {turn['ai']}")
        
        # Add current input
        context_parts.append(f"Human: {current_input}")
        context_parts.append("AI:")
        
        context = ' '.join(context_parts)
        return context
    
    def generate_response(self, user_input: str, max_length: int = 100, 
                         temperature: float = 0.8, top_p: float = 0.9) -> str:
        """
        Generate a conversational response
        
        Args:
            user_input: User's question or statement
            max_length: Maximum response length
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated response string
        """
        # Build context with conversation history
        context = self._build_context(user_input)
        
        # Tokenize context
        tokens = self._tokenize(context)
        
        # Truncate if too long
        if len(tokens) > self.max_context_length:
            tokens = tokens[-self.max_context_length:]
        
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        # Generate response
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions
                logits, _ = self.model(input_tensor, None)
                
                # Get logits for last token
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # Stop if EOS or padding
                if next_token in [self.EOS_IDX, self.PAD_IDX]:
                    break
                
                generated_tokens.append(next_token)
                
                # Update input for next iteration
                input_tensor = torch.cat([
                    input_tensor,
                    torch.tensor([[next_token]], dtype=torch.long).to(self.device)
                ], dim=1)
                
                # Truncate if exceeding context window
                if input_tensor.size(1) > self.max_context_length:
                    input_tensor = input_tensor[:, -self.max_context_length:]
        
        # Decode generated tokens
        response = self._detokenize(generated_tokens)
        
        # Post-process response
        response = self._postprocess_response(response)
        
        # Update conversation history
        self.conversation_history.append({
            'human': user_input,
            'ai': response
        })
        
        return response
    
    def _postprocess_response(self, response: str) -> str:
        """Clean up and format the generated response"""
        # Remove repeated phrases
        words = response.split()
        cleaned_words = []
        for i, word in enumerate(words):
            # Check for repetitions (same 3-word sequence)
            if i > 5:
                last_three = ' '.join(words[i-3:i])
                current_three = ' '.join(words[i:i+3]) if i+3 <= len(words) else ''
                if last_three == current_three:
                    break
            cleaned_words.append(word)
        
        response = ' '.join(cleaned_words)
        
        # Capitalize first letter
        if response:
            response = response[0].upper() + response[1:]
        
        # Ensure proper sentence ending
        if response and not response[-1] in ['.', '!', '?']:
            response += '.'
        
        return response
    
    def explain_concept(self, concept: str, detail_level: str = 'simple') -> str:
        """
        Explain an AI concept at different detail levels
        
        Args:
            concept: AI concept to explain (e.g., "neural network", "backpropagation")
            detail_level: 'simple', 'intermediate', or 'technical'
            
        Returns:
            Explanation string
        """
        prompts = {
            'simple': f"Explain {concept} in simple terms for a high school student",
            'intermediate': f"Explain {concept} with moderate detail",
            'technical': f"Provide a technical explanation of {concept}"
        }
        
        prompt = prompts.get(detail_level, prompts['simple'])
        return self.generate_response(prompt, max_length=150, temperature=0.7)
    
    def answer_question(self, question: str) -> str:
        """
        Answer a specific question about AI
        
        Args:
            question: Question to answer
            
        Returns:
            Answer string
        """
        return self.generate_response(question, max_length=120, temperature=0.75)
    
    def propose_question(self) -> str:
        """
        Propose an interesting question to continue the conversation
        
        Returns:
            A question to discuss
        """
        prompt = "What's an interesting question about AI we could explore?"
        return self.generate_response(prompt, max_length=50, temperature=0.9)
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation so far"""
        if not self.conversation_history:
            return "No conversation yet."
        
        summary = f"Conversation History ({len(self.conversation_history)} turns):\n"
        for i, turn in enumerate(self.conversation_history, 1):
            summary += f"\n[Turn {i}]\n"
            summary += f"  Human: {turn['human'][:100]}...\n"
            summary += f"  AI: {turn['ai'][:100]}...\n"
        
        return summary
    
    def save_conversation(self, filepath: str):
        """Save conversation history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        print(f"💾 Conversation saved to {filepath}")
    
    def load_conversation(self, filepath: str):
        """Load conversation history from file"""
        with open(filepath, 'r') as f:
            self.conversation_history = json.load(f)
        print(f"📂 Conversation loaded from {filepath}")


class PodcastHost:
    """
    Specialized agent for podcast hosting
    Manages multi-turn conversations and topic flow
    """
    
    def __init__(self, agent: ConversationalAgent):
        """
        Initialize podcast host
        
        Args:
            agent: Underlying conversational agent
        """
        self.agent = agent
        self.current_topic = None
        self.topics_covered = []
    
    def introduce_topic(self, topic: str) -> str:
        """
        Introduce a new podcast topic
        
        Args:
            topic: Topic to introduce
            
        Returns:
            Introduction statement
        """
        self.current_topic = topic
        prompt = f"Introduce the topic of {topic} for our AI literacy podcast"
        return self.agent.generate_response(prompt, max_length=80, temperature=0.7)
    
    def ask_follow_up_question(self, previous_answer: str) -> str:
        """
        Generate a follow-up question based on previous answer
        
        Args:
            previous_answer: The previous response
            
        Returns:
            Follow-up question
        """
        prompt = f"Based on what we just discussed about {self.current_topic}, what's a good follow-up question?"
        return self.agent.generate_response(prompt, max_length=40, temperature=0.8)
    
    def wrap_up_topic(self) -> str:
        """
        Wrap up the current topic
        
        Returns:
            Wrap-up statement
        """
        if self.current_topic:
            self.topics_covered.append(self.current_topic)
            prompt = f"Summarize the key points about {self.current_topic}"
            response = self.agent.generate_response(prompt, max_length=60, temperature=0.7)
            self.current_topic = None
            return response
        return "Let's move on to the next topic."
    
    def get_episode_summary(self) -> str:
        """Get summary of topics covered in this episode"""
        if not self.topics_covered:
            return "No topics covered yet."
        
        summary = "Topics covered in this episode:\n"
        for i, topic in enumerate(self.topics_covered, 1):
            summary += f"{i}. {topic}\n"
        
        return summary
