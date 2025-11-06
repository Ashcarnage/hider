# 🎮 HIDER Agent - Fine-tuned Gemma 3 270M

Fine-tune Gemma 3 270M to play Hide-and-Seek using **Modal** (serverless GPU platform with free credits)!

## 🚀 Quick Start

```bash
# 1. Install Modal
pip install modal

# 2. Setup Modal account (get free credits!)
modal setup

# 3. Train the model on cloud GPU
modal run train_modal.py

# 4. Test the model
modal run test_model.py
```

**That's it!** Training takes ~20 minutes on a free T4 GPU.

## 📖 What is this?

This project fine-tunes Google's **Gemma 3 270M** model to control a HIDER agent in a Hide-and-Seek game. The model learns to:
- Process 112-dimensional observation vectors
- Output optimal actions: `[move_x, move_y, move_z, lock, grab]`
- Make strategic decisions in 3D space

## 🎯 Features

- ✅ **Modal Integration**: Serverless GPU training with free credits
- ✅ **Unsloth Optimization**: 2x faster training, 60% less memory
- ✅ **LoRA Fine-tuning**: Efficient parameter-efficient training
- ✅ **2000 Training Examples**: Real Hide-and-Seek game data
- ✅ **Automated Testing**: Built-in model evaluation
- ✅ **Production Ready**: Export merged 16-bit models

## 📊 Dataset

**Source**: `hider_raw.jsonl` (2000 examples)

**Format**:
```json
{
  "messages": [
    {"role": "system", "content": "You are a HIDER agent..."},
    {"role": "user", "content": "[4.232, 1.478, 0.000, ...]"},
    {"role": "assistant", "content": "[8, 0, 7, 1, 1]"}
  ]
}
```

## 💰 Cost

**Using Modal's free tier ($30 credits):**
- Training: ~$0.40 per run (20-30 mins on T4)
- Testing: ~$0.05 per run (2-3 mins)
- **You can train 75+ times with free credits!**

## 📚 Documentation

- **[SETUP.md](SETUP.md)** - Detailed setup guide
- **[train_modal.py](train_modal.py)** - Training script with comments
- **[test_model.py](test_model.py)** - Testing script with samples

## 🛠️ Tech Stack

- **Model**: Gemma 3 270M (Google)
- **Cloud**: Modal (serverless GPU)
- **Optimization**: Unsloth (2x speedup)
- **Method**: LoRA fine-tuning
- **Framework**: Transformers, TRL, PEFT

## 🎓 Learn More

- [Modal Documentation](https://modal.com/docs)
- [Gemma 3 Model Card](https://huggingface.co/google/gemma-3-270m-it)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)

---

**Made with ❤️ using Modal's free GPU credits**