# Model Failure Diagnosis

## The Problem
After 11 epochs of training, the model produces nonsensical outputs like:
- "What is Impressionism?" → "the cochlea"
- "Who was Vincent van Gogh?" → "man and woman... born in the car"

## Root Causes Identified

### 1. **Model is Learning Generic Word Patterns, Not Semantic Content**

**Evidence:**
- When asked "Q: What is Impressionism?\nA:", the top predicted tokens are:
  - "the" (score: 6.01)
  - "an" (score: 5.55)
  - "a" (score: 3.71)

These are just the most common English words! The model hasn't learned ANY art knowledge.

### 2. **Training Dataset is Too Diverse for a Small Model**

Your training mix:
- **Art Knowledge**: WikiArt templates, synthetic descriptions
- **AI Literacy**: ELI5, SciQ, TruthfulQA
- **Conversational**: OpenAssistant, Anthropic HH

**Problem:** A 35M parameter model trained from scratch is TOO SMALL to learn:
- Art history AND
- AI concepts AND
- General conversation

It's spreading its capacity too thin and learning NOTHING deeply.

### 3. **Training Data Quality Issues**

Looking at your WikiArt templates:
```python
"Q: What defines {style} art?\nA: {style} is an artistic movement characterized by {artist}'s approach to {genre}."
```

**Problems:**
- Templates are repetitive and formulaic
- Model learns the TEMPLATE pattern, not the content
- Not enough diverse, natural language about each concept

### 4. **No Instruction Tuning**

The model is trained as a pure language model (next-token prediction) on mixed data.
It never learned that:
- "Q:" means "answer this question"
- It should generate relevant answers
- It should stay on topic

## Why Perplexity Looks OK But Outputs Are Garbage

- **Perplexity: 7.65** seems reasonable
- But this just means the model can predict common words ("the", "a", "an", "is")
- It doesn't mean it learned semantic meaning!

## The Fix: Three Options

### Option 1: **Simplify & Focus (FASTEST - 4 hours)**
- Train ONLY on art knowledge
- Remove AI literacy and conversation datasets
- Use more diverse, natural text (not templates)
- This will make it an art specialist, not a general chatbot

### Option 2: **Use Pretrained Model (BEST - 2 hours)**
- Start with distilgpt2 (already knows English)
- Fine-tune on your art dataset
- Model 2 approach - this is WHY pretrained models exist!

### Option 3: **Fix From Scratch (SLOW - 20+ hours)**
- Need 10x more art data
- Better data curation
- Longer training (50+ epochs)
- Probably still won't be as good as Option 2

## Recommended Solution

**Use Option 2: Fine-tune DistilGPT-2**

Why:
1. DistilGPT-2 already knows English grammar and common knowledge
2. You just need to teach it art-specific facts
3. Will work in 5-10 epochs instead of 50+
4. This is literally what Model 2 is supposed to demonstrate!

The whole point of your coursework comparison is:
- **Model 1 (from scratch)**: Hard, needs massive data, slow
- **Model 2 (pretrained)**: Easy, needs less data, fast, better results

Your Model 1 is proving the point that training from scratch is HARD! 😅

Would you like me to help you:
A) Fix Model 1 properly (Option 1 - simpler data)
B) Move to Model 2 (pretrained fine-tuning)
C) Both?
