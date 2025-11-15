# Neural Canvas - Art Expert LLM

## Quick Start

### 1. Install Dependencies
```bash
pip install torch transformers datasets tqdm numpy matplotlib
```

### 2. Train Model 1 (Custom - 35-56M params)
```bash
cd llm/scripts
python train_custom_model.py
```

### 3. Fine-tune Model 2 (Pretrained - 82M params)
```bash
cd llm/scripts
python finetune_pretrained_model.py
```

### 4. Chat with Models
```bash
cd llm/scripts
python chat_with_models.py --model both
```

### 5. Evaluate
```bash
cd llm/scripts
python evaluate_models.py
```

## Model Sizes

- **Small**: 32M parameters (6 layers, 384 dim)
- **Base**: 56M parameters (8 layers, 512 dim) ← Default
- **Large**: 91M parameters (10 layers, 640 dim)

## Files

```
llm/
├── models/art_expert_model.py          # Custom transformer
├── utils/curated_art_dataset.py        # Dataset loader
└── scripts/
    ├── train_custom_model.py           # Train Model 1
    ├── finetune_pretrained_model.py    # Fine-tune Model 2
    ├── chat_with_models.py             # Interactive chat
    └── evaluate_models.py              # Benchmarking
```

## Dataset (Streaming - No Download)

- **Art Knowledge**: 80K samples (WikiArt, descriptions)
- **AI Literacy**: 75K samples (ELI5, SciQ, TruthfulQA)
- **Conversational**: 80K samples (OpenAssistant, Anthropic HH)
- **Total**: ~235K curated samples

All datasets stream from HuggingFace - no disk space needed!
