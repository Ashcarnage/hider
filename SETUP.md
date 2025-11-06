# 🎮 HIDER Agent Fine-tuning with Modal

Fine-tune Gemma 3 270M on Hide-and-Seek game dataset using **Modal** - a serverless GPU platform with free credits!

## 🌟 Why Modal?

- **Free GPU Credits**: Get started with free credits for GPU compute
- **Serverless**: No infrastructure management needed
- **T4 GPU Access**: Perfect for fine-tuning small models like Gemma 3 270M
- **Easy Scaling**: Scale to multiple GPUs when needed

## 📋 Prerequisites

1. **Python 3.11+**
2. **Modal Account** (free signup at https://modal.com)
3. **HuggingFace Account** (optional, for model access)

## 🚀 Quick Start

### Step 1: Install Modal

```bash
pip install modal
```

### Step 2: Set Up Modal Account

```bash
# This will open a browser for authentication
modal setup
```

You'll be prompted to:
1. Create a Modal account (or login)
2. Authenticate your CLI
3. Get your free credits!

### Step 3: (Optional) Set Up HuggingFace Token

If you want to access gated models or push to HuggingFace Hub:

```bash
# Create a HuggingFace secret in Modal
modal secret create huggingface-secret HF_TOKEN=your_token_here
```

Get your token from: https://huggingface.co/settings/tokens

### Step 4: Verify Your Setup

```bash
modal token new
```

This should show your Modal account is connected.

## 🏋️ Training the Model

### Run Fine-tuning on Modal

```bash
modal run train_modal.py
```

This will:
- ✅ Upload your dataset to Modal
- ✅ Spin up a T4 GPU instance
- ✅ Install all dependencies (Unsloth, Transformers, etc.)
- ✅ Fine-tune Gemma 3 270M with LoRA
- ✅ Save the model to Modal's persistent storage
- ✅ Shut down the GPU automatically

**Expected time**: 15-30 minutes on T4 GPU

**Cost**: Uses your free credits! (~$0.40 worth with free tier)

### Monitor Training

While training runs, you can:
- View logs in your terminal
- Check Modal dashboard: https://modal.com/dashboard
- Close terminal - job continues in cloud!

## 🧪 Testing the Model

### Option 1: Test on Modal (Recommended)

```bash
modal run test_model.py
```

This will:
- Load your fine-tuned model from Modal storage
- Run inference on test samples
- Display predictions vs expected outputs
- Show accuracy metrics

### Option 2: Test Locally

First, download the model from Modal:

```bash
# Download model from Modal volume
modal volume get hider-models /models/hider_sft ./models/hider_sft
```

Then run local tests:

```bash
python test_model.py --local --model-path ./models/hider_sft/final_model_merged_16bit
```

**Note**: Local testing requires a GPU. Modal testing is recommended!

## 📊 Understanding the Dataset

The `hider_raw.jsonl` dataset contains 2000 examples of:

**Input**: Observation vector (112 numbers representing game state)
```
[4.232, 1.478, 0.000, 0.000, 0.000, ...]
```

**Output**: Action vector (5 numbers)
```
[move_x, move_y, move_z, lock, grab]
```

Where:
- `move_x, move_y, move_z`: Movement in 3D space (0-10)
- `lock`: Lock object (0 or 1)
- `grab`: Grab object (0 or 1)

## 🎯 Model Architecture

- **Base Model**: Gemma 3 270M Instruction-tuned
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Optimization**: Unsloth for 2x faster training
- **LoRA Rank**: 16
- **Training**: 15 epochs, batch size 4, gradient accumulation 8

## 💰 Modal Credits & Pricing

### Free Tier
- **$30 free credits** when you sign up
- T4 GPU: **~$0.026/minute** ($1.56/hour)
- This training job: **~$0.40** (15-30 mins)
- **You can run ~75 training jobs** with free credits!

### Tips to Save Credits
1. Use `timeout` parameter to auto-stop long jobs
2. Test on small dataset first
3. Use `modal volume` for persistent storage (free)
4. Monitor dashboard to track spending

## 📁 Project Structure

```
hider/
├── train_modal.py          # Main training script (runs on Modal)
├── test_model.py            # Model testing script
├── hider_raw.jsonl          # Training dataset (2000 examples)
├── Hide_and_Seek (2).ipynb  # Original Colab notebook
├── requirements.txt         # Python dependencies
├── SETUP.md                 # This file
└── README.md                # Project overview
```

## 🔧 Configuration

Edit these in `train_modal.py`:

```python
TRAINING_CONFIG = {
    "EPOCHS": 15,              # Number of training epochs
    "BATCH_SIZE": 4,           # Batch size per GPU
    "LEARNING_RATE": 2e-4,     # Learning rate
    "LORA_R": 16,              # LoRA rank
    "MAX_SEQ_LENGTH": 2048,    # Max sequence length
}
```

## 🐛 Troubleshooting

### "modal: command not found"
```bash
pip install --upgrade modal
```

### "Authentication required"
```bash
modal setup
```

### "GPU out of memory"
- Reduce `BATCH_SIZE` in config
- Reduce `MAX_SEQ_LENGTH`
- Enable 4-bit quantization: `load_in_4bit=True`

### "Volume not found"
```bash
# List all volumes
modal volume list

# Create volume manually
modal volume create hider-models
```

### Training too slow
- Switch to A10G GPU: change `gpu="T4"` to `gpu="A10G"` (costs more)
- Reduce dataset size for testing
- Ensure you're using Unsloth (already configured)

## 📚 Additional Resources

- **Modal Docs**: https://modal.com/docs
- **Modal Examples**: https://github.com/modal-labs/llm-finetuning
- **Unsloth Docs**: https://docs.unsloth.ai
- **Gemma 3 Model Card**: https://huggingface.co/google/gemma-3-270m-it

## 🎉 Next Steps

After fine-tuning:

1. **Test the model** with `modal run test_model.py`
2. **Deploy for inference** (create a Modal web endpoint)
3. **Export to ONNX** for faster inference
4. **Push to HuggingFace Hub** for sharing
5. **Try different hyperparameters** to improve accuracy

## 💡 Pro Tips

1. **Start with a small test**: Reduce dataset to 100 examples first
2. **Use Modal dashboard**: Monitor GPU usage and costs
3. **Save checkpoints**: Enabled by default every 100 steps
4. **Version your experiments**: Use different volume names
5. **Share your models**: Push to HF Hub after training

## 🤝 Contributing

Have improvements? Found bugs? PRs welcome!

## 📝 License

This project uses:
- **Gemma 3**: Licensed under Gemma Terms of Use
- **Unsloth**: Apache 2.0
- **Modal**: Subject to Modal's terms of service

---

**Happy Fine-tuning! 🚀**

Questions? Check the [Modal Discord](https://discord.gg/modal) or [GitHub Issues](https://github.com/modal-labs/modal-examples/issues)
