# 🎨 Neural Canvas

**Modern Language Model Training Framework**  
*Building state-of-the-art transformers from scratch with 2025 best practices*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🚀 What is Neural Canvas?

Neural Canvas is a comprehensive deep learning project that implements **modern transformer architectures from scratch** and compares them against fine-tuned pre-trained models. Think of it as your personal AI training ground where you can experiment with cutting-edge NLP techniques without the black box.

### Why This Project Exists

Most people use pre-trained models without understanding what's under the hood. This project bridges that gap by:

- ✨ **Building a Llama-style transformer from scratch** (RoPE, RMSNorm, GQA, SwiGLU)
- 🔬 **Training on real-world datasets** (500K+ text samples)
- 📊 **Comparing custom models vs fine-tuned giants** (Fair benchmarking)
- 🌱 **Tracking sustainability** (Carbon emissions, energy usage)
- 🎓 **Learning by doing** (Educational but production-ready)

---

## 🏗️ Architecture Overview

### Model 1: Modern Transformer (Built from Scratch)

A 1.2B parameter transformer implementing 2025's best practices:

| Feature | What It Does | Why It Matters |
|---------|-------------|----------------|
| **RoPE** | Rotary Position Embeddings | Better position awareness than absolute embeddings |
| **RMSNorm** | Root Mean Square Normalization | More stable training than LayerNorm |
| **GQA** | Grouped Query Attention | Faster inference, less memory than MHA |
| **SwiGLU** | Gated Linear Units | Better expressiveness than standard FFN |

**Specs:**
- 🧠 **12 Layers**, 16 attention heads (4 KV heads)
- 📏 **1024 hidden dimensions**, 2048 max sequence length
- 📦 **500K training samples** from OpenWebText + C4
- ⏱️ **30 epochs** with learning rate warmup and gradient clipping

### Model 2: Fine-Tuned Llama 2 (Baseline Comparison)

- 🦙 **Llama 2 7B** with 4-bit quantization (LoRA adapters)
- 🎯 **Fine-tuned** on conversational datasets
- ⚡ **Efficient training** with PEFT (Parameter-Efficient Fine-Tuning)

---

## 📚 Datasets

We train on large-scale, diverse text corpora:

| Dataset | Size | Purpose | License |
|---------|------|---------|---------|
| **OpenWebText** | 250K samples | General web knowledge | Open |
| **C4** | 250K samples | Clean crawled web text | Open |
| **WikiArt** | Optional | Creative descriptions | CC-BY |

**Total:** 500K+ high-quality training examples

---

## 🛠️ Project Structure

```
neural-canvas/
├── llm/
│   ├── models/
│   │   ├── modern_transformer.py    # Our transformer implementation
│   │   ├── transformer_model.py     # Base transformer
│   │   └── conversational_agent.py  # Inference interface
│   ├── scripts/
│   │   ├── train_model1.py          # Train from scratch
│   │   ├── train_model2.py          # Fine-tune Llama 2
│   │   └── inference.py             # Chat with models
│   └── utils/
│       ├── dataset_loader.py        # Data pipeline
│       ├── model_evaluator.py       # Metrics & benchmarks
│       └── sustainability_tracker.py # Carbon tracking
├── train.csh                        # Training launcher (csh)
├── run_model1.csh                   # Quick start script
├── requirements.txt                 # Python dependencies
├── TRAINING.md                      # Detailed training guide
└── README.md                        # You are here!
```

---

## ⚡ Quick Start

### 1️⃣ **Setup Environment**

```bash
# Clone the repo
git clone https://github.com/MuhammadMaazA/neural-canvas.git
cd neural-canvas

# Create virtual environment
python3 -m venv venv
source venv/bin/activate.csh  # for csh users

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

### 2️⃣ **Train Model 1 (From Scratch)**

```csh
# Set cache directories (optional, for storage management)
setenv HF_HOME /cs/student/projects1/2023/muhamaaz/datasets
setenv HF_DATASETS_CACHE /cs/student/projects1/2023/muhamaaz/datasets

# Start training
cd llm/scripts
python train_model1.py
```

**Expected:**
- ⏱️ **Training time:** 8-12 hours on A100 / 12-18 hours on RTX 3090 Ti
- 💾 **Checkpoints:** Saved to `/checkpoints/model1_best.pt` (~800MB)
- 📊 **Logs:** Real-time training stats in `/logs/`

### 3️⃣ **Train Model 2 (Fine-tune Llama)**

```csh
cd llm/scripts
python train_model2.py
```

**Expected:**
- ⏱️ **Training time:** 4-6 hours on A100
- 💾 **Memory:** ~16GB GPU with 4-bit quantization
- 📦 **LoRA adapters:** ~50MB (not the full 7B model!)

### 4️⃣ **Chat with Your Model**

```csh
cd llm/scripts
python inference.py --model model1
```

---

## 🎯 Training in Background

Long training sessions? No problem! Keep your model training even after logout:

### Option 1: Using `nohup` (Simplest)

```csh
nohup ./train.csh >& training.log &
echo $! > training.pid

# Monitor progress
tail -f training.log

# Stop training
kill `cat training.pid`
```

### Option 2: Using `screen` (Recommended)

```csh
# Start screen session
screen -S training

# Run training
./train.csh

# Detach: Press Ctrl+A then D
# Reattach later: screen -r training
```

### Option 3: Using `tmux`

```csh
tmux new -s training
./train.csh
# Detach: Press Ctrl+B then D
# Reattach: tmux attach -t training
```

---

## 📊 Monitoring & Metrics

### GPU Usage

```bash
watch -n 1 nvidia-smi
```

### Training Metrics

We track:
- 📉 **Loss** (cross-entropy)
- 🎯 **Perplexity** (lower is better)
- 🌍 **Carbon emissions** (via CodeCarbon)
- ⚡ **Energy consumption** (kWh)
- ⏱️ **Training speed** (samples/sec)

All metrics are logged to TensorBoard:

```bash
tensorboard --logdir logs/
```

---

## 🧪 Evaluation & Benchmarks

### Automated Evaluation

```csh
cd llm/scripts
python -c "from utils.model_evaluator import evaluate_model; evaluate_model('model1')"
```

Metrics include:
- **ROUGE scores** (overlap with reference text)
- **BLEU scores** (translation quality)
- **Perplexity** (confidence of predictions)
- **Human evaluation** (manual quality checks)

---

## 🌱 Sustainability Tracking

We care about the environment! Every training run tracks:

- 🌍 **CO₂ emissions** (kg CO₂eq)
- ⚡ **Energy consumed** (kWh)
- 🌳 **Trees needed to offset** (estimated)

Results saved to `logs/emissions.csv`

---

## 🎓 Learning Resources

### Key Papers Implemented

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - RMSNorm
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - SwiGLU
- [Llama 2: Open Foundation Models](https://arxiv.org/abs/2307.09288) - GQA

### Tutorials

See `TRAINING.md` for detailed guides on:
- Storage management (handling large datasets)
- Checkpoint recovery (resume training)
- Hyperparameter tuning
- Debugging common errors

---

## 🤝 Contributing

This is a learning project, but contributions are welcome! Feel free to:

- 🐛 Report bugs via Issues
- 💡 Suggest features
- 🔧 Submit pull requests
- 📖 Improve documentation

---

## 📝 License

MIT License - See `LICENSE` file for details.

---

## 👤 Author

**Muhammad Maaz**  
📧 mmaaz2005@hotmail.com  
🐙 GitHub: [@MuhammadMaazA](https://github.com/MuhammadMaazA)

---

## 🙏 Acknowledgments

- 🎓 Built for academic exploration and learning
- 🤗 Hugging Face for transformers library
- 🔥 PyTorch team for the framework
- 🌍 CodeCarbon for sustainability tracking
- 💻 UCL CS Department for compute resources

---

**Happy Training! 🚀**

*"The best way to understand transformers is to build one yourself."*
