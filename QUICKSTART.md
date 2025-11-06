# ⚡ Quick Start Guide - Run in 5 Minutes!

Follow these steps exactly to train your HIDER model:

## 🎯 Step-by-Step Instructions

### Step 1: Navigate to Project Directory

```bash
cd /path/to/hider
```

Replace `/path/to/hider` with where you cloned this repo.

### Step 2: Run Setup Script (EASIEST WAY)

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- ✅ Install Modal
- ✅ Set up your account (opens browser)
- ✅ Get your $30 free credits
- ✅ Verify everything is ready

**OR do it manually:**

```bash
# Install Modal
pip install modal

# Setup account (opens browser)
modal setup

# Verify it worked
modal token show
```

### Step 3: Train the Model 🏋️

```bash
modal run train_modal.py
```

**What happens:**
- Uploads dataset to Modal
- Starts a T4 GPU in the cloud
- Installs all dependencies automatically
- Fine-tunes Gemma 3 270M (15 epochs)
- Saves model to cloud storage
- Shuts down GPU automatically

**Time**: 20-30 minutes
**Cost**: ~$0.40 (uses your free credits!)

**You'll see output like:**
```
🚀 Launching fine-tuning job on Modal...
📤 Uploading dataset to Modal...
🦥 LOADING GEMMA 3 270M WITH UNSLOTH
✅ Model loaded successfully!
🎯 APPLYING LORA
...
🏋️ STARTING TRAINING
Epoch 1/15: [▓▓▓▓▓▓░░░░] 60% | Loss: 0.234
...
✅ TRAINING COMPLETE!
💾 SAVING MODEL
✅ Model saved to Modal volume
```

**Pro tip**: You can close your terminal - the job continues in the cloud!

### Step 4: Test the Model 🧪

```bash
modal run test_model.py
```

**What happens:**
- Loads your fine-tuned model from cloud
- Runs 3 test samples
- Shows predictions vs expected outputs
- Displays accuracy metrics

**Time**: 2-3 minutes
**Cost**: ~$0.05

**You'll see output like:**
```
======================================================================
TEST 1: Sample 1: Initial observation
======================================================================

📊 Observation (truncated):
   [4.232, 1.478, 0.000, 0.000, 0.000, ...]

🎯 Expected Output:
   [8, 0, 7, 1, 1]

🤖 Model Prediction:
   [8, 0, 7, 1, 1]

✅ Output Format: Valid
```

### Step 5: Monitor Your Usage 📊

Visit: https://modal.com/dashboard

You can see:
- Running jobs
- GPU usage
- Credits remaining
- Job logs

## 🚨 Troubleshooting

### "modal: command not found"

```bash
pip install --upgrade modal
# Or if using conda:
conda install -c conda-forge modal
```

### "Authentication required"

```bash
modal setup
# Follow the browser prompts
```

### "Dataset not found"

Make sure you're in the correct directory:
```bash
ls -la
# You should see: hider_raw.jsonl
```

### "GPU out of memory"

Edit `train_modal.py` and reduce batch size:
```python
"BATCH_SIZE": 2,  # Change from 4 to 2
```

## 🎓 What Each File Does

| File | Purpose | When to Run |
|------|---------|-------------|
| `setup.sh` | Install and configure Modal | Once (first time) |
| `train_modal.py` | Train the model | When you want to fine-tune |
| `test_model.py` | Test the model | After training completes |

## 💡 Common Workflows

### First Time Setup
```bash
./setup.sh                    # Setup Modal
modal run train_modal.py      # Train model (~30 mins)
modal run test_model.py       # Test model (~3 mins)
```

### Re-train with Different Settings
1. Edit `train_modal.py` (change EPOCHS, BATCH_SIZE, etc.)
2. Run: `modal run train_modal.py`

### Test Existing Model
```bash
modal run test_model.py
```

## 📞 Need Help?

- **Check detailed guide**: `SETUP.md`
- **View training code**: `train_modal.py` (well commented)
- **Modal docs**: https://modal.com/docs
- **Modal Discord**: https://discord.gg/modal

## 🎉 You're Ready!

Run this command now:
```bash
modal run train_modal.py
```

Grab a coffee ☕ and watch your model train in the cloud! 🚀
