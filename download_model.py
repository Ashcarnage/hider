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

    # List available files in volume
    print("📋 Checking available files in Modal volume...")
    for item in volume.listdir("/models/hider_sft"):
        print(f"   - {item}")
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
