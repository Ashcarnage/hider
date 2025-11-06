"""
Test Fine-tuned Gemma 3 270M HIDER Model
=========================================

This script tests the fine-tuned model on sample observations and displays
the predicted actions in a formatted way.

Usage:
    1. Local testing (if model downloaded):
       python test_model.py --local --model-path ./models/hider_sft/final_model_merged_16bit

    2. Modal testing (recommended):
       modal run test_model.py
"""

import modal
import json
import argparse
from typing import List, Dict

# ============================================================================
# MODAL CONFIGURATION
# ============================================================================

app = modal.App("hider-model-test")

# Same image as training
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "peft==0.13.2",
        "trl==0.12.1",
        "bitsandbytes==0.44.1",
        "scipy",
    )
    .run_commands(
        "pip install --upgrade --no-deps --force-reinstall git+https://github.com/unslothai/unsloth.git"
    )
)

# Access the saved model volume
volume = modal.Volume.from_name("hider-models", create_if_missing=True)

# ============================================================================
# TEST DATA SAMPLES
# ============================================================================

TEST_SAMPLES = [
    {
        "description": "Sample 1: Initial observation",
        "observation": "[4.232, 1.478, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.480, 5.023, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.466, 4.454, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 3.240, 2.032, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 5.056, 2.095, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 4.227, 1.803, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 4.007, 0.285, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.468, 1.501, 1.147, 0.298, 0.235, 0.202, 0.184, 0.176, 0.176, 0.184, 0.202, 1.841, 1.063, 1.022, 1.260, 1.232, 1.260, 1.349, 2.515, 1.989, 1.707, 1.045, 0.999, 0.999, 1.555, 1.707, 1.989, 1.815, 1.607, 1.501]",
        "expected": "[8, 0, 7, 1, 1]"
    },
    {
        "description": "Sample 2: Mid-game observation",
        "observation": "[3.240, 2.032, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 4.232, 1.478, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.480, 5.023, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.466, 4.454, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 5.056, 2.095, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 4.227, 1.803, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 4.007, 0.285, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.837, 1.703, 2.134, 3.041, 1.168, 1.002, 0.912, 0.873, 0.873, 0.776, 0.479, 0.358, 0.296, 0.262, 0.245, 0.240, 0.245, 0.262, 0.296, 0.358, 0.479, 0.776, 2.043, 2.043, 2.137, 1.787, 2.735, 1.063, 0.916, 0.856]",
        "expected": "[3, 5, 6, 1, 1]"
    },
    {
        "description": "Sample 3: Complex observation",
        "observation": "[4.357, 1.293, -0.002, 0.267, 2.803, -4.245, -0.489, 3.600, 1.000, 0.031, 0.480, 5.023, -0.000, 0.000, 0.000, -0.000, 0.013, 0.000, 0.000, 0.031, 1.466, 4.454, -0.000, 0.000, 0.000, -0.000, 0.013, 0.000, 0.000, 0.031, 3.180, 2.032, -0.000, 0.133, -1.367, 0.000, -0.086, 1.800, 1.000, 0.031, 0.000, 0.000, 0.000, 5.056, 2.095, -0.000, 1.000, 0.000, 0.000, -0.000, 0.013, -0.000, 0.000, 0.000, 0.000, 4.227, 1.803, -0.000, 1.000, 0.000, -0.000, 0.000, 0.013, 0.000, 1.000, 1.000, 1.000, 0.000, 0.000, 4.007, 0.285, -0.000, 1.000, 0.000, 0.000, 0.000, -0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.343, 1.373, 1.470, 1.109, 0.877, 0.416, 0.379, 0.362, 0.362, 0.379, 0.416, 2.028, 1.270, 1.485, 1.387, 1.357, 1.387, 1.485, 2.200, 1.740, 0.931, 0.847, 0.810, 1.300, 1.360, 1.493, 1.740, 1.660, 1.470, 1.373]",
        "expected": "[2, 0, 1, 1, 1]"
    },
]

# ============================================================================
# INFERENCE FUNCTION (MODAL)
# ============================================================================

@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": volume},
    timeout=600,
)
def run_inference(test_samples: List[Dict], model_path: str = "/models/hider_sft/final_model_merged_16bit"):
    """
    Run inference on test samples using the fine-tuned model.
    """
    import torch
    from unsloth import FastLanguageModel

    print("="*70)
    print("🔍 LOADING FINE-TUNED MODEL FOR TESTING")
    print("="*70)

    # Load the fine-tuned model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )

    # Set model to inference mode
    FastLanguageModel.for_inference(model)

    print(f"✅ Model loaded from: {model_path}")
    print(f"✅ Model ready for inference\n")

    # System prompt
    system_prompt = "You are a HIDER agent. Given observation vector, output exactly 5 numbers: [move_x, move_y, move_z, lock, grab] where move values are 0-10, lock and grab are 0 or 1."

    results = []

    print("="*70)
    print("🧪 RUNNING TESTS")
    print("="*70 + "\n")

    for i, sample in enumerate(test_samples, 1):
        print(f"{'='*70}")
        print(f"TEST {i}: {sample['description']}")
        print(f"{'='*70}")

        # Create prompt
        prompt = f"{system_prompt}\n\n{sample['observation']}\n\n"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,  # Low temperature for deterministic output
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the answer (after the observation)
        answer = generated_text.split(sample['observation'])[-1].strip()

        # Display results
        print(f"\n📊 Observation (truncated):")
        print(f"   {sample['observation'][:100]}...")

        print(f"\n🎯 Expected Output:")
        print(f"   {sample['expected']}")

        print(f"\n🤖 Model Prediction:")
        print(f"   {answer}")

        # Simple validation check
        is_valid = answer.startswith('[') and answer.endswith(']')
        if is_valid:
            print(f"\n✅ Output Format: Valid")
        else:
            print(f"\n❌ Output Format: Invalid (expected format: [x, y, z, lock, grab])")

        print(f"\n{'='*70}\n")

        results.append({
            "test_number": i,
            "description": sample['description'],
            "expected": sample['expected'],
            "predicted": answer,
            "is_valid_format": is_valid,
        })

    return results


# ============================================================================
# LOCAL INFERENCE (FOR DOWNLOADED MODELS)
# ============================================================================

def run_inference_local(model_path: str, test_samples: List[Dict]):
    """
    Run inference locally if you have the model downloaded.
    """
    import torch
    from unsloth import FastLanguageModel

    print("="*70)
    print("🔍 LOADING FINE-TUNED MODEL (LOCAL)")
    print("="*70)

    try:
        # Load the fine-tuned model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,
        )

        # Set model to inference mode
        FastLanguageModel.for_inference(model)

        print(f"✅ Model loaded from: {model_path}")
        print(f"✅ Model ready for inference\n")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Tip: Make sure you've downloaded the model or use Modal testing instead")
        print("   Run: modal run test_model.py")
        return

    # System prompt
    system_prompt = "You are a HIDER agent. Given observation vector, output exactly 5 numbers: [move_x, move_y, move_z, lock, grab] where move values are 0-10, lock and grab are 0 or 1."

    print("="*70)
    print("🧪 RUNNING TESTS")
    print("="*70 + "\n")

    for i, sample in enumerate(test_samples, 1):
        print(f"{'='*70}")
        print(f"TEST {i}: {sample['description']}")
        print(f"{'='*70}")

        # Create prompt
        prompt = f"{system_prompt}\n\n{sample['observation']}\n\n"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text.split(sample['observation'])[-1].strip()

        # Display results
        print(f"\n📊 Observation (truncated):")
        print(f"   {sample['observation'][:100]}...")

        print(f"\n🎯 Expected Output:")
        print(f"   {sample['expected']}")

        print(f"\n🤖 Model Prediction:")
        print(f"   {answer}")

        is_valid = answer.startswith('[') and answer.endswith(']')
        if is_valid:
            print(f"\n✅ Output Format: Valid")
        else:
            print(f"\n❌ Output Format: Invalid")

        print(f"\n{'='*70}\n")


# ============================================================================
# MODAL ENTRYPOINT
# ============================================================================

@app.local_entrypoint()
def main():
    """
    Main entry point for Modal testing.
    Run with: modal run test_model.py
    """
    print("🚀 Running model tests on Modal...")
    print("")

    # Run inference remotely
    results = run_inference.remote(TEST_SAMPLES)

    # Summary
    print("\n" + "="*70)
    print("📈 TEST SUMMARY")
    print("="*70)

    valid_count = sum(1 for r in results if r['is_valid_format'])
    total_count = len(results)

    print(f"\nTotal tests: {total_count}")
    print(f"Valid format outputs: {valid_count}/{total_count}")
    print(f"Success rate: {(valid_count/total_count)*100:.1f}%")

    print("\n✅ Testing complete!")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test fine-tuned HIDER model")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally instead of on Modal"
    )
    parser.add_argument(
        "--model-path",
        type="str",
        default="./models/hider_sft/final_model_merged_16bit",
        help="Path to the fine-tuned model"
    )

    args = parser.parse_args()

    if args.local:
        print("🏠 Running local inference...")
        run_inference_local(args.model_path, TEST_SAMPLES)
    else:
        print("💡 Tip: Run with 'modal run test_model.py' to test on Modal's GPUs")
        print("   Or use --local flag to test locally if you have the model downloaded")
