# 🧪 Testing Your Fine-tuned Model

Quick guide to download and test your trained HIDER model.

## 📥 Step 1: Download Model from Modal

```bash
# Download the trained model to your local machine
modal run download_model.py
```

**What this does:**
- Downloads LoRA adapters to `./models/hider_sft_lora/`
- Downloads merged model to `./models/hider_sft_merged/`
- Takes 2-5 minutes depending on internet speed

**Expected output:**
```
📥 DOWNLOADING FINE-TUNED MODEL FROM MODAL
📁 Local directory: ./models
☁️  Remote path: /models/hider_sft
⬇️  Downloading models...
✅ LoRA adapters downloaded!
✅ Merged model downloaded!
🎉 DOWNLOAD COMPLETE!
```

---

## 🧪 Step 2: Test the Model Locally

### Install Dependencies (if needed)

```bash
pip install torch transformers unsloth datasets
```

### Run Tests

```bash
# Test with default model path
python test_model_local.py

# Or specify custom path
python test_model_local.py --model-path ./models/hider_sft_merged
```

**What this does:**
- Loads your fine-tuned model
- Runs 5 test cases with different game scenarios
- Shows predictions vs expected outputs
- Calculates accuracy and gives you a grade!

---

## 📊 Understanding Test Results

### Example Output:

```
======================================================================
TEST 1/5: Test 1: Initial game state
======================================================================

📊 Observation (first 50 values):
   [4.232, 1.478, 0.000, 0.000, 0.000, ...]

🎯 Expected Output:
   [8, 0, 7, 1, 1]

🤖 Model Prediction:
   [8, 0, 7, 1, 1]

✅ Output Format: Valid
   Parsed: [8, 0, 7, 1, 1]

✅ EXACT MATCH! 🎉
```

### Summary Metrics:

- **Valid format outputs**: Did the model output correct format?
- **Exact matches**: How many predictions were 100% correct?
- **Average accuracy**: Overall accuracy across all tests
- **Grade**: A+ to D based on performance

### Grading Scale:

| Grade | Exact Matches | Meaning |
|-------|---------------|---------|
| A+ | 5/5 (100%) | Perfect! 🌟 |
| A | 4/5 (80%+) | Excellent! 🎉 |
| B | 3/5 (60%+) | Good! 👍 |
| C | 2/5 (40%+) | Fair 😊 |
| D | 0-1/5 | Needs improvement 🔧 |

---

## 🔧 Troubleshooting

### "Model not found"

Make sure you downloaded it first:
```bash
modal run download_model.py
```

### "CUDA out of memory"

Your GPU doesn't have enough memory. Try:

**Option 1: Use CPU**
```python
# Edit test_model_local.py, change:
load_in_4bit=True  # Add this line
```

**Option 2: Use 4-bit quantization**
Already done in the code if Unsloth is installed!

### "ModuleNotFoundError: No module named 'unsloth'"

Install Unsloth:
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

Or the test will automatically fall back to standard Transformers.

---

## 🎯 What Actions Mean

The model outputs 5 numbers: `[move_x, move_y, move_z, lock, grab]`

- **move_x, move_y, move_z**: Movement in 3D space (0-10)
  - Example: `[8, 0, 7]` = move 8 units in X, 0 in Y, 7 in Z
- **lock**: Lock object in place (0 or 1)
  - `1` = lock, `0` = don't lock
- **grab**: Grab/hold object (0 or 1)
  - `1` = grab, `0` = release

### Example Action:
```
[8, 0, 7, 1, 1]
 │  │  │  │  └─ Grab object
 │  │  │  └──── Lock object
 │  │  └─────── Move 7 units in Z
 │  └────────── No movement in Y
 └───────────── Move 8 units in X
```

---

## 🚀 Advanced: Test on Custom Data

Create your own test samples:

```python
# Add to TEST_SAMPLES in test_model_local.py
{
    "description": "My custom test",
    "observation": "[1.0, 2.0, 3.0, ...]",  # Your 112 values
    "expected": "[5, 5, 5, 0, 1]"  # Expected action
}
```

---

## 📈 Improving Model Performance

If your model isn't performing well:

1. **Train longer**: Increase `EPOCHS` in `train_modal.py`
2. **More data**: Add more examples to `hider_raw.jsonl`
3. **Adjust learning rate**: Try different values
4. **Larger model**: Try Gemma 1B or 3B instead of 270M

---

## 💡 Quick Commands Reference

```bash
# Download model from Modal
modal run download_model.py

# Test locally
python test_model_local.py

# Test with custom model path
python test_model_local.py --model-path ./my_model

# Check model files
ls -lh ./models/hider_sft_merged/
```

---

**Happy Testing! 🎉**
