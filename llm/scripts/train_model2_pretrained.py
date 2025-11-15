"""
Model 2: Fine-tune Pretrained Model (For Comparison)
====================================================
Uses a better pretrained model than GPT-2

Coursework requirement: Compare with Model 1 (trained from scratch)

Available models:
- GPT-2 Large (774M params) - Better than base GPT-2
- Phi-2 (2.7B params) - Microsoft's efficient model
- GPT-2 XL (1.5B params) - Largest GPT-2
"""

import os
os.environ['HF_HOME'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['HF_DATASETS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/cs/student/projects1/2023/muhamaaz/datasets'

import torch
from torch.utils.data import DataLoader
import time
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.art_ai_dataset_loader import load_art_ai_literacy_datasets


class Model2Config:
    """Configuration for pretrained model fine-tuning"""

    # Choose your model (pick ONE):
    # MODEL_NAME = "gpt2-large"           # 774M params - Good balance
    MODEL_NAME = "gpt2-xl"              # 1.5B params - Best GPT-2
    # MODEL_NAME = "microsoft/phi-2"      # 2.7B params - Very good but larger

    # Dataset sizes (smaller for fine-tuning)
    ART_KNOWLEDGE_SAMPLES = 50000
    AI_LITERACY_SAMPLES = 80000
    CONVERSATIONAL_SAMPLES = 40000
    IMAGE_CAPTIONS_SAMPLES = 10000
    # Total: ~180K samples

    # Fine-tuning hyperparameters
    NUM_EPOCHS = 3              # Few epochs for fine-tuning
    BATCH_SIZE = 4              # Smaller for large model
    GRADIENT_ACCUMULATION = 8   # Effective batch = 32
    LEARNING_RATE = 2e-5        # Very low for fine-tuning
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    MAX_LEN = 1024

    TRAIN_SPLIT = 0.95
    VAL_SPLIT = 0.05

    OUTPUT_DIR = "/cs/student/projects1/2023/muhamaaz/checkpoints/model2_pretrained"
    LOG_DIR = "/cs/student/projects1/2023/muhamaaz/logs"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_logging(log_dir: str):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"model2_pretrained_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def prepare_dataset(texts, tokenizer, max_len):
    """Prepare dataset for HuggingFace Trainer"""

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_len,
            padding='max_length'
        )

    # Create HuggingFace dataset
    dataset_dict = {'text': texts}
    dataset = Dataset.from_dict(dataset_dict)

    # Tokenize
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )

    return tokenized


def main():
    """Main fine-tuning function"""
    config = Model2Config()
    logger = setup_logging(config.LOG_DIR)

    print("=" * 80)
    print("MODEL 2: PRETRAINED MODEL FINE-TUNING")
    print("=" * 80)
    print("Coursework: COMP0220 - Comparison Model")
    print(f"Base Model: {config.MODEL_NAME}")
    print("=" * 80)

    logger.info(f"Starting Model 2 fine-tuning with {config.MODEL_NAME}")

    # Load pretrained model and tokenizer
    logger.info(f"Loading pretrained model: {config.MODEL_NAME}")
    print(f"\nLoading {config.MODEL_NAME}...")
    print("This may take a few minutes to download...")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float16 if config.DEVICE == 'cuda' else torch.float32,
        device_map='auto' if config.DEVICE == 'cuda' else None
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Model loaded")
    print(f"  Total params: {total_params/1e9:.2f}B" if total_params > 1e9 else f"  Total params: {total_params/1e6:.1f}M")
    print(f"  Trainable: {trainable_params/1e9:.2f}B" if trainable_params > 1e9 else f"  Trainable: {trainable_params/1e6:.1f}M")
    logger.info(f"Model parameters: {total_params/1e6:.1f}M")

    # Load datasets
    logger.info("Loading Art + AI Literacy datasets...")
    print("\nLoading datasets...")

    all_texts = load_art_ai_literacy_datasets(
        art_knowledge_samples=config.ART_KNOWLEDGE_SAMPLES,
        ai_literacy_samples=config.AI_LITERACY_SAMPLES,
        conversational_samples=config.CONVERSATIONAL_SAMPLES,
        image_captions_samples=config.IMAGE_CAPTIONS_SAMPLES
    )

    logger.info(f"Loaded {len(all_texts):,} samples")

    # Split datasets
    split_idx = int(len(all_texts) * config.TRAIN_SPLIT)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]

    logger.info(f"Train: {len(train_texts):,}, Val: {len(val_texts):,}")
    print(f"Train: {len(train_texts):,}, Val: {len(val_texts):,}")

    # Prepare datasets
    print("\nTokenizing datasets...")
    train_dataset = prepare_dataset(train_texts, tokenizer, config.MAX_LEN)
    val_dataset = prepare_dataset(val_texts, tokenizer, config.MAX_LEN)

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=config.WARMUP_STEPS,
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=3,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True if config.DEVICE == 'cuda' else False,
        report_to="none",  # Disable wandb
        logging_dir=config.LOG_DIR,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "=" * 80)
    print("STARTING FINE-TUNING")
    print("=" * 80)
    print(f"This will take approximately {config.NUM_EPOCHS * len(train_texts) // (config.BATCH_SIZE * config.GRADIENT_ACCUMULATION) // 60:.1f} hours")
    print("=" * 80 + "\n")

    logger.info("Starting fine-tuning...")
    start_time = time.time()

    trainer.train()

    total_time = time.time() - start_time
    logger.info(f"Fine-tuning complete: {total_time/3600:.2f} hours")

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(os.path.join(config.OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(config.OUTPUT_DIR, "final"))

    print("\n" + "=" * 80)
    print("FINE-TUNING COMPLETE!")
    print("=" * 80)
    print(f"Model saved to: {config.OUTPUT_DIR}/final")
    print(f"Training time: {total_time/3600:.2f} hours")
    print("=" * 80)

    logger.info("Model 2 fine-tuning complete")


if __name__ == "__main__":
    main()
