"""
Download Fine-tuned Model from Modal Volume
============================================

This script downloads the trained model from Modal's cloud storage
to your local machine.

Usage:
    modal run download_model.py
"""

import modal

app = modal.App("download-hider-model")

# Access the same volume where model was saved
volume = modal.Volume.from_name("hider-models")

@app.local_entrypoint()
def main():
    """
    Download the fine-tuned model to local directory.
    """
    import os

    print("="*70)
    print("📥 DOWNLOADING FINE-TUNED MODEL FROM MODAL")
    print("="*70)
    print()

    # Create local directory for models
    local_model_dir = "./models"
    os.makedirs(local_model_dir, exist_ok=True)

    print(f"📁 Local directory: {local_model_dir}")
    print(f"☁️  Remote path: /models/hider_sft")
    print()

    # Download the merged 16-bit model (ready for inference)
    print("Downloading merged 16-bit model...")
    print("(This is the model ready for inference)")
    print()

    # List available files in volume (check root first)
    print("📋 Checking available files in Modal volume...")
    try:
        # First, check root directory
        print("   Checking volume root...")
        root_files = volume.listdir("/")
        print(f"   Root contents: {root_files}")

        # Then check models directory
        if any("models" in str(f) for f in root_files):
            print("   Checking /models directory...")
            models_files = volume.listdir("/models")
            print(f"   Models contents: {models_files}")

            # Check hider_sft directory
            if any("hider_sft" in str(f) for f in models_files):
                print("   Checking /models/hider_sft directory...")
                hider_files = volume.listdir("/models/hider_sft")
                for item in hider_files:
                    print(f"      - {item}")
            else:
                print("   ⚠️  hider_sft directory not found in /models")
        else:
            print("   ⚠️  models directory not found in volume root")
    except Exception as e:
        print(f"   ⚠️  Error listing files: {e}")
        print("   The model might not have been saved during training.")
        print()
        print("💡 Possible solutions:")
        print("   1. Check if training actually completed successfully")
        print("   2. The volume might be empty - try training again")
        print("   3. Check Modal dashboard: https://modal.com/storage")
        return
    print()

    # Download both versions
    print("⬇️  Downloading models (this may take a few minutes)...")
    print()

    # Download LoRA adapters
    print("1. Downloading LoRA adapters...")
    volume.copy_files(
        "/models/hider_sft/final_model",
        local_model_dir + "/hider_sft_lora"
    )
    print("   ✅ LoRA adapters downloaded!")

    # Download merged model
    print("2. Downloading merged 16-bit model...")
    volume.copy_files(
        "/models/hider_sft/final_model_merged_16bit",
        local_model_dir + "/hider_sft_merged"
    )
    print("   ✅ Merged model downloaded!")

    print()
    print("="*70)
    print("🎉 DOWNLOAD COMPLETE!")
    print("="*70)
    print()
    print(f"📁 Models saved to:")
    print(f"   LoRA adapters: {local_model_dir}/hider_sft_lora/")
    print(f"   Merged model:  {local_model_dir}/hider_sft_merged/")
    print()
    print("💡 You can now use these models for inference!")
    print()
