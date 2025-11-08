# LLM Component

UCL COMP0220 Deep Learning Coursework

## Models

**Model 1: Transformer (From Scratch)**
- Architecture: RoPE, RMSNorm, GQA, SwiGLU
- Parameters: 200M
- Datasets: OpenWebText, C4, WikiArt (1M+ samples)

**Model 2: Llama 2 7B Fine-Tuning**
- Method: LoRA with 4-bit quantization
- Trainable: 16M parameters

## Structure

```
llm/
├── models/
│   ├── modern_transformer.py
│   ├── transformer_model.py
│   └── conversational_agent.py
├── utils/
│   ├── dataset_loader.py
│   ├── model_evaluator.py
│   └── sustainability_tracker.py
└── scripts/
    ├── train_model1.py
    ├── train_model2.py
    └── inference.py
```

## Installation

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

## Training

```bash
# Model 1 (from scratch)
cd scripts && python train_model1.py

# Model 2 (fine-tuning)
cd scripts && python train_model2.py
```

## Datasets

1. OpenWebText: 500K samples
2. C4: 500K samples
3. WikiArt: 100K art descriptions

Total: 1.1M training samples

## Coursework Requirements

- Model 1: Custom transformer from scratch
- Model 2: Llama 2 fine-tuned with LoRA
- Three datasets from HuggingFace
- Regularization, evaluation metrics, sustainability tracking
