"""
UCL COMP0220 Coursework - Model 2: Mistral 7B Fine-tuning
Pre-trained model for comparison
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset

print("="*80)
print("MODEL 2: Mistral 7B Fine-tuning")
print("="*80)

ai_data = [
    {"text": "<s>[INST] What is AI? [/INST] AI is when computers learn and solve problems like humans.</s>"},
    {"text": "<s>[INST] How do computers learn? [/INST] By analyzing data and finding patterns.</s>"},
    {"text": "<s>[INST] What is a neural network? [/INST] A system inspired by brains with connected layers.</s>"},
] * 1000

print(f"Data: {len(ai_data)} samples")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./mistral_out",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_steps=200,
    report_to="none",
)

trainer = SFTTrainer(model=model, train_dataset=Dataset.from_list(ai_data), tokenizer=tokenizer, args=training_args, peft_config=lora_config, dataset_text_field="text", max_seq_length=512)

print("Training...")
trainer.train()

model.save_pretrained("./MODEL2_FINAL")
tokenizer.save_pretrained("./MODEL2_FINAL")
print("COMPLETE! Saved to MODEL2_FINAL")

