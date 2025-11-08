# UCL COMP0220 Deep Learning Coursework - LLM Component

## Final Implementation: LLM vs LLM Comparison

### Model 1: GPT-2 Style Transformer FROM SCRATCH
- Architecture: 12-layer Transformer (124M parameters)
- Datasets: Cornell (100K) + ELI5 (15K) + Tech QA (15K) = 130K samples
- Training: 6-8 hours on RTX 3090 Ti
- Expected Perplexity: 20-30

### Model 2: Mistral 7B (Fine-tuned)
- Pre-trained: 7B parameters
- Fine-tuned on: ELI5 dataset
- Training: 30-45 minutes
- Expected Perplexity: 10-15

## Datasets (3 Sources - Coursework Requirement)

### 1. Cornell Movie Dialogs
- Using: 100,000 conversation pairs
- Purpose: General dialogue patterns
- Auto-downloaded: Yes

### 2. ELI5 (Explain Like I'm 5)
- Using: 15,000 samples
- Purpose: AI literacy explanations
- Built-in: Yes

### 3. Technical Deep Learning Q&A
- Using: 15,000 samples
- Purpose: Technical accuracy
- Built-in: Yes

Total for Model 1: 130,000 training samples

## Training on UCL GPUs

### Connect to RTX 3090 Ti
```bash
ssh -J muhamaaz@knuckles.cs.ucl.ac.uk muhamaaz@canada-l.cs.ucl.ac.uk
cd /cs/student/projects1/2023/muhamaaz/DeepLearningCoursework
```

### Train Model 1 (6-8 hours)
```bash
python3 COMPLETE_LLM_TRAINING.py
```

### Train Model 2 (30-45 min)
```bash
pip3 install --user trl peft bitsandbytes
python3 COMPLETE_MISTRAL_FINETUNE.py
```

## Key Files

- COMPLETE_LLM_TRAINING.py - Model 1 (Transformer from scratch)
- COMPLETE_MISTRAL_FINETUNE.py - Model 2 (Mistral fine-tuned)
- llm/transformer_model.py - Transformer architecture
- llm/dataset_utils.py - Dataset loading
- requirements.txt - Dependencies

## Coursework Requirements Met

- Two LLM models (Transformer from scratch + Mistral fine-tuned)
- 3+ datasets for Model 1
- AI literacy focus
- Regularization techniques
- Fair LLM vs LLM comparison
- Professional grade (130K samples, 30 epochs)
- FREE resources only
