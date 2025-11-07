"""
Test Fine-tuned HIDER Model Locally
====================================

This script tests the downloaded fine-tuned model on sample observations
and displays predictions in a nice format.

Requirements:
    pip install torch transformers unsloth datasets

Usage:
    python test_model_local.py

    # Or specify custom model path:
    python test_model_local.py --model-path ./models/hider_sft_merged
"""

import argparse
import json
import torch
from typing import List, Dict

# ============================================================================
# TEST DATA SAMPLES
# ============================================================================

TEST_SAMPLES = [
    {
        "description": "Test 1: Initial game state",
        "observation": "[4.232, 1.478, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.480, 5.023, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.466, 4.454, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 3.240, 2.032, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 5.056, 2.095, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 4.227, 1.803, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 4.007, 0.285, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.468, 1.501, 1.147, 0.298, 0.235, 0.202, 0.184, 0.176, 0.176, 0.184, 0.202, 1.841, 1.063, 1.022, 1.260, 1.232, 1.260, 1.349, 2.515, 1.989, 1.707, 1.045, 0.999, 0.999, 1.555, 1.707, 1.989, 1.815, 1.607, 1.501]",
        "expected": "[8, 0, 7, 1, 1]"
    },
    {
        "description": "Test 2: Mid-game positioning",
        "observation": "[3.240, 2.032, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 4.232, 1.478, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.480, 5.023, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.466, 4.454, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 5.056, 2.095, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 4.227, 1.803, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 4.007, 0.285, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.837, 1.703, 2.134, 3.041, 1.168, 1.002, 0.912, 0.873, 0.873, 0.776, 0.479, 0.358, 0.296, 0.262, 0.245, 0.240, 0.245, 0.262, 0.296, 0.358, 0.479, 0.776, 2.043, 2.043, 2.137, 1.787, 2.735, 1.063, 0.916, 0.856]",
        "expected": "[3, 5, 6, 1, 1]"
    },
    {
        "description": "Test 3: Complex movement scenario",
        "observation": "[4.357, 1.293, -0.002, 0.267, 2.803, -4.245, -0.489, 3.600, 1.000, 0.031, 0.480, 5.023, -0.000, 0.000, 0.000, -0.000, 0.013, 0.000, 0.000, 0.031, 1.466, 4.454, -0.000, 0.000, 0.000, -0.000, 0.013, 0.000, 0.000, 0.031, 3.180, 2.032, -0.000, 0.133, -1.367, 0.000, -0.086, 1.800, 1.000, 0.031, 0.000, 0.000, 0.000, 5.056, 2.095, -0.000, 1.000, 0.000, 0.000, -0.000, 0.013, -0.000, 0.000, 0.000, 0.000, 4.227, 1.803, -0.000, 1.000, 0.000, -0.000, 0.000, 0.013, 0.000, 1.000, 1.000, 1.000, 0.000, 0.000, 4.007, 0.285, -0.000, 1.000, 0.000, 0.000, 0.000, -0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.343, 1.373, 1.470, 1.109, 0.877, 0.416, 0.379, 0.362, 0.362, 0.379, 0.416, 2.028, 1.270, 1.485, 1.387, 1.357, 1.387, 1.485, 2.200, 1.740, 0.931, 0.847, 0.810, 1.300, 1.360, 1.493, 1.740, 1.660, 1.470, 1.373]",
        "expected": "[2, 0, 1, 1, 1]"
    },
    {
        "description": "Test 4: Object grab scenario",
        "observation": "[3.180, 2.032, -0.000, 0.133, -1.367, 0.000, -0.086, 1.800, 1.000, 0.031, 4.357, 1.293, -0.002, 0.267, 2.803, -4.245, -0.489, 3.600, 1.000, 0.031, 0.480, 5.023, -0.000, 0.000, 0.000, -0.000, 0.013, 0.000, 0.000, 0.031, 1.466, 4.454, -0.000, 0.000, 0.000, -0.000, 0.013, 0.000, 0.000, 0.031, 1.000, 0.000, 0.000, 5.056, 2.095, -0.000, 1.000, 0.000, 0.000, -0.000, 0.013, -0.000, 0.000, 0.000, 0.000, 4.227, 1.803, -0.000, 1.000, 0.000, -0.000, 0.000, 0.013, 0.000, 0.000, 1.000, 1.000, 0.000, 0.000, 4.007, 0.285, -0.000, 1.000, 0.000, 0.000, 0.000, -0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.897, 1.764, 2.134, 3.115, 1.168, 1.002, 0.912, 0.873, 0.873, 0.581, 0.359, 0.268, 0.222, 0.197, 0.184, 0.180, 0.184, 0.197, 0.222, 0.268, 0.359, 0.581, 2.043, 2.043, 2.137, 1.786, 2.735, 1.270, 2.759, 0.917]",
        "expected": "[5, 8, 6, 1, 1]"
    },
    {
        "description": "Test 5: Advanced movement",
        "observation": "[4.377, 0.893, -0.002, -0.264, -1.454, -6.302, -0.798, -7.200, 1.000, 0.062, 0.480, 5.023, -0.000, 0.000, -0.000, 0.000, 0.001, 0.000, 0.000, 0.062, 1.466, 4.454, -0.000, 0.000, -0.000, 0.000, 0.001, 0.000, 0.000, 0.062, 3.158, 2.140, 0.001, 0.279, 0.064, 2.285, 0.300, 1.800, 1.000, 0.062, 0.000, 0.000, 0.000, 5.056, 2.095, -0.000, 1.000, 0.000, 0.000, -0.000, 0.001, -0.000, 0.000, 0.000, 0.000, 4.227, 1.803, -0.000, 1.000, 0.000, -0.000, 0.000, 0.001, -0.000, 1.000, 1.000, 1.000, 0.000, 0.000, 4.007, 0.285, -0.000, 1.000, 0.000, 0.000, 0.000, -0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.323, 1.352, 1.448, 1.635, 1.416, 1.215, 2.111, 0.765, 0.765, 0.800, 2.318, 1.606, 1.703, 1.508, 1.408, 1.377, 4.293, 2.195, 1.519, 0.546, 0.468, 0.426, 0.408, 0.898, 0.939, 1.031, 1.201, 1.519, 1.448, 1.352]",
        "expected": "[5, 1, 7, 1, 0]"
    },
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_action(text: str) -> Dict:
    """
    Parse model output to extract action values.
    """
    # Try to find a list pattern [x, y, z, lock, grab]
    import re
    pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    match = re.search(pattern, text)

    if match:
        return {
            "raw": text,
            "parsed": [int(match.group(i)) for i in range(1, 6)],
            "move_x": int(match.group(1)),
            "move_y": int(match.group(2)),
            "move_z": int(match.group(3)),
            "lock": int(match.group(4)),
            "grab": int(match.group(5)),
            "valid": True
        }
    else:
        return {
            "raw": text,
            "parsed": None,
            "valid": False
        }

def compare_actions(predicted: List[int], expected: str) -> Dict:
    """
    Compare predicted action with expected action.
    """
    expected_list = json.loads(expected)

    if predicted is None:
        return {"match": False, "exact_match": False, "partial_matches": 0}

    exact_match = predicted == expected_list
    partial_matches = sum(p == e for p, e in zip(predicted, expected_list))

    return {
        "match": exact_match,
        "exact_match": exact_match,
        "partial_matches": partial_matches,
        "accuracy": partial_matches / 5 * 100
    }

# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def test_model(model_path: str, test_samples: List[Dict]):
    """
    Test the fine-tuned model on sample observations.
    """
    print("="*70)
    print("🔍 LOADING FINE-TUNED MODEL")
    print("="*70)
    print()

    try:
        from unsloth import FastLanguageModel

        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,
        )

        # Set to inference mode
        FastLanguageModel.for_inference(model)

        print(f"✅ Model loaded from: {model_path}")
        print(f"✅ Device: {next(model.parameters()).device}")
        print()

    except Exception as e:
        print(f"❌ Error loading model with Unsloth: {e}")
        print()
        print("💡 Trying with standard Transformers...")
        print()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        print(f"✅ Model loaded from: {model_path}")
        print()

    # System prompt
    system_prompt = "You are a HIDER agent. Given observation vector, output exactly 5 numbers: [move_x, move_y, move_z, lock, grab] where move values are 0-10, lock and grab are 0 or 1."

    # Run tests
    results = []
    correct = 0
    total = len(test_samples)

    print("="*70)
    print("🧪 RUNNING TESTS")
    print("="*70)
    print()

    for i, sample in enumerate(test_samples, 1):
        print(f"{'='*70}")
        print(f"TEST {i}/{total}: {sample['description']}")
        print(f"{'='*70}")
        print()

        # Create prompt
        prompt = f"{system_prompt}\n\n{sample['observation']}\n\n"

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        print("🤖 Generating prediction...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,  # Low temperature for deterministic output
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer
        answer = generated_text.split(sample['observation'])[-1].strip()

        # Parse action
        parsed = parse_action(answer)

        # Compare with expected
        comparison = compare_actions(parsed['parsed'], sample['expected'])

        # Display results
        print(f"\n📊 Observation (first 50 values):")
        obs_values = sample['observation'].split(', ')[:10]
        print(f"   [{', '.join(obs_values)}, ...]")

        print(f"\n🎯 Expected Output:")
        print(f"   {sample['expected']}")

        print(f"\n🤖 Model Prediction:")
        print(f"   {answer}")

        if parsed['valid']:
            print(f"\n✅ Output Format: Valid")
            print(f"   Parsed: {parsed['parsed']}")

            if comparison['exact_match']:
                print(f"\n✅ EXACT MATCH! 🎉")
                correct += 1
            else:
                print(f"\n⚠️  Partial Match")
                print(f"   Accuracy: {comparison['accuracy']:.1f}%")
                print(f"   Matching values: {comparison['partial_matches']}/5")
        else:
            print(f"\n❌ Output Format: Invalid")

        print(f"\n{'='*70}\n")

        results.append({
            "test_number": i,
            "description": sample['description'],
            "expected": sample['expected'],
            "predicted": answer,
            "parsed": parsed,
            "comparison": comparison,
        })

    # Summary
    print("="*70)
    print("📈 TEST SUMMARY")
    print("="*70)
    print()

    valid_outputs = sum(1 for r in results if r['parsed']['valid'])
    exact_matches = sum(1 for r in results if r['comparison'].get('exact_match', False))

    avg_accuracy = sum(r['comparison'].get('accuracy', 0) for r in results) / total

    print(f"Total tests: {total}")
    print(f"Valid format outputs: {valid_outputs}/{total} ({valid_outputs/total*100:.1f}%)")
    print(f"Exact matches: {exact_matches}/{total} ({exact_matches/total*100:.1f}%)")
    print(f"Average accuracy: {avg_accuracy:.1f}%")
    print()

    # Grade
    if exact_matches == total:
        grade = "A+ PERFECT! 🌟"
    elif exact_matches >= total * 0.8:
        grade = "A Excellent! 🎉"
    elif exact_matches >= total * 0.6:
        grade = "B Good! 👍"
    elif exact_matches >= total * 0.4:
        grade = "C Fair 😊"
    else:
        grade = "D Needs improvement 🔧"

    print(f"Grade: {grade}")
    print()

    print("✅ Testing complete!")
    print()

    return results

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test fine-tuned HIDER model locally")
    parser.add_argument(
        "--model-path",
        type="str",
        default="./models/hider_sft_merged",
        help="Path to the fine-tuned model directory"
    )

    args = parser.parse_args()

    print("🎮 HIDER Model Testing")
    print()

    # Check if model exists
    import os
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found at: {args.model_path}")
        print()
        print("💡 Did you download the model?")
        print("   Run: modal run download_model.py")
        print()
        exit(1)

    # Run tests
    results = test_model(args.model_path, TEST_SAMPLES)
