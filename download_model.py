"""
Download Fine-tuned Model from Modal Volume
============================================

This script downloads the trained model from Modal's cloud storage
to your local machine.

Usage:
    modal run download_model.py
"""

import modal
import os

app = modal.App("download-hider-model")

# Access the same volume where model was saved
volume = modal.Volume.from_name("hider-models")

@app.local_entrypoint()
def main():
    """
    Download the fine-tuned model to local directory.
    """
    print("="*70)
    print("📥 DOWNLOADING FINE-TUNED MODEL FROM MODAL")
    print("="*70)
    print()

    # Create local directory for models
    local_model_dir = "./models"
    os.makedirs(local_model_dir, exist_ok=True)

    print(f"📁 Local directory: {local_model_dir}")
    print(f"☁️  Remote path: /hider_sft")
    print()

    # List available files in volume
    print("📋 Checking available files in Modal volume...")
    try:
        for item in volume.listdir("/hider_sft"):
            print(f"   - {item}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        print("   Model files not found in volume!")
        return
    print()

    # Download both versions
    print("⬇️  Downloading models (this may take a few minutes)...")
    print()

    # Use absolute paths
    lora_dir = os.path.abspath(os.path.join(local_model_dir, "hider_sft_lora"))
    merged_dir = os.path.abspath(os.path.join(local_model_dir, "hider_sft_merged"))

    print(f"   LoRA destination: {lora_dir}")
    print(f"   Merged destination: {merged_dir}")
    print()

    # Download LoRA adapters
    print("1. Downloading LoRA adapters...")
    try:
        # Modal's copy_files will create the destination directory
        volume.copy_files(
            "/hider_sft/final_model",  # Source in Modal volume (no trailing slash)
            lora_dir                    # Destination on local machine
        )
        print("   ✅ LoRA adapters downloaded!")
    except Exception as e:
        print(f"   ⚠️  Error downloading LoRA adapters: {e}")

    # Download merged model
    print("2. Downloading merged 16-bit model...")
    try:
        volume.copy_files(
            "/hider_sft/final_model_merged_16bit",  # Source in Modal volume
            merged_dir                               # Destination on local machine
        )
        print("   ✅ Merged model downloaded!")
    except Exception as e:
        print(f"   ⚠️  Error downloading merged model: {e}")

    print()
    print("="*70)
    print("🎉 DOWNLOAD COMPLETE!")
    print("="*70)
    print()
    print(f"📁 Models saved to:")
    print(f"   LoRA adapters: {lora_dir}/")
    print(f"   Merged model:  {merged_dir}/")
    print()
    print("💡 You can now use these models for inference!")
    print("   Run: python test_model_local.py")
    print()
