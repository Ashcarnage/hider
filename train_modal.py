"""
Fine-tune Gemma 3 270M on Hide-and-Seek Dataset using Modal
============================================================

This script uses Modal (GPU cloud provider) to fine-tune the Gemma 3 270M model
on the hide-and-seek dataset. Modal provides serverless GPU access with free credits.

Setup:
1. Install Modal: pip install modal
2. Set up Modal account: modal setup
3. Run this script: modal run train_modal.py
"""

import modal
import os

# ============================================================================
# MODAL CONFIGURATION
# ============================================================================

# Create Modal app
app = modal.App("hider-gemma3-finetune")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # Install git for installing Unsloth from GitHub
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

# Create a persistent volume to store the trained model
volume = modal.Volume.from_name("hider-models", create_if_missing=True)

# Training configuration
TRAINING_CONFIG = {
    "DATA_FILE": "hider_raw.jsonl",
    "MODEL_NAME": "unsloth/gemma-3-270m-it",
    "OUTPUT_DIR": "/models/hider_sft",
    "MAX_SEQ_LENGTH": 2048,
    "EPOCHS": 15,
    "BATCH_SIZE": 4,
    "GRADIENT_ACCUMULATION_STEPS": 8,
    "LEARNING_RATE": 2e-4,
    "LORA_R": 16,
    "LORA_ALPHA": 16,
    "WARMUP_STEPS": 50,
    "SAVE_STEPS": 100,
    "EVAL_STEPS": 100,
}

# ============================================================================
# FINE-TUNING FUNCTION
# ============================================================================

@app.function(
    image=image,
    gpu="T4",  # Use T4 GPU (free tier friendly)
    volumes={"/models": volume},
    timeout=3600 * 3,  # 3 hour timeout
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def finetune_model():
    """
    Fine-tune Gemma 3 270M using Unsloth for memory-efficient training.
    """
    import torch
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    import json

    print("="*70)
    print("🚀 STARTING GEMMA 3 270M FINE-TUNING ON MODAL")
    print("="*70)

    # Load configuration
    config = TRAINING_CONFIG

    # ========================================================================
    # STEP 1: LOAD MODEL AND TOKENIZER
    # ========================================================================

    print("\n" + "="*70)
    print("🦥 LOADING GEMMA 3 270M WITH UNSLOTH")
    print("="*70)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["MODEL_NAME"],
        max_seq_length=config["MAX_SEQ_LENGTH"],
        dtype=None,  # Auto-detect best dtype
        load_in_4bit=False,  # Use 16-bit for better quality on small model
    )

    print(f"✅ Model loaded successfully!")
    print(f"   Model: {config['MODEL_NAME']}")
    print(f"   Max sequence length: {config['MAX_SEQ_LENGTH']}")

    # ========================================================================
    # STEP 2: APPLY LORA
    # ========================================================================

    print("\n" + "="*70)
    print("🎯 APPLYING LORA")
    print("="*70)

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["LORA_R"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config["LORA_ALPHA"],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
    )

    print(f"✅ LoRA applied successfully!")
    print(f"   LoRA rank: {config['LORA_R']}")
    print(f"   LoRA alpha: {config['LORA_ALPHA']}")

    # ========================================================================
    # STEP 3: LOAD AND PREPARE DATASET
    # ========================================================================

    print("\n" + "="*70)
    print("📊 LOADING DATASET")
    print("="*70)

    # Load JSONL file from local directory
    dataset = load_dataset('json', data_files=config["DATA_FILE"], split='train')
    print(f"✅ Loaded {len(dataset)} examples")

    # Format dataset
    def format_example(example):
        messages = example['messages']

        # Combine all messages into training text
        text = f"{messages[0]['content']}\n\n"  # System
        text += f"{messages[1]['content']}\n\n"  # User
        text += f"{messages[2]['content']}"  # Assistant

        # Add EOS token
        text += tokenizer.eos_token

        return {"text": text}

    print("   Formatting examples...")
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # Split into train/eval (90/10)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    print(f"✅ Dataset prepared:")
    print(f"   Training examples: {len(train_dataset)}")
    print(f"   Evaluation examples: {len(eval_dataset)}")

    # ========================================================================
    # STEP 4: CONFIGURE TRAINING
    # ========================================================================

    print("\n" + "="*70)
    print("⚙️ CONFIGURING TRAINING")
    print("="*70)

    training_args = TrainingArguments(
        output_dir=config["OUTPUT_DIR"],
        num_train_epochs=config["EPOCHS"],
        per_device_train_batch_size=config["BATCH_SIZE"],
        per_device_eval_batch_size=config["BATCH_SIZE"],
        gradient_accumulation_steps=config["GRADIENT_ACCUMULATION_STEPS"],
        learning_rate=config["LEARNING_RATE"],
        weight_decay=0.01,
        logging_steps=10,
        save_steps=config["SAVE_STEPS"],
        eval_steps=config["EVAL_STEPS"],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=config["WARMUP_STEPS"],
        lr_scheduler_type="cosine",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        report_to="none",
        push_to_hub=False,
        save_total_limit=3,
    )

    effective_batch_size = (
        config["BATCH_SIZE"] * config["GRADIENT_ACCUMULATION_STEPS"]
    )
    print(f"✅ Training configuration:")
    print(f"   Epochs: {config['EPOCHS']}")
    print(f"   Batch size: {config['BATCH_SIZE']}")
    print(f"   Gradient accumulation: {config['GRADIENT_ACCUMULATION_STEPS']}")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Learning rate: {config['LEARNING_RATE']}")

    # ========================================================================
    # STEP 5: CREATE TRAINER
    # ========================================================================

    print("\n" + "="*70)
    print("🚀 INITIALIZING TRAINER")
    print("="*70)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=config["MAX_SEQ_LENGTH"],
        packing=False,
    )

    print("✅ Trainer initialized!")

    # ========================================================================
    # STEP 6: TRAIN THE MODEL
    # ========================================================================

    print("\n" + "="*70)
    print("🏋️ STARTING TRAINING")
    print("="*70)
    print("Estimated time: 15-30 minutes on T4 GPU")
    print("="*70 + "\n")

    # Start training
    trainer.train()

    print("\n✅ TRAINING COMPLETE!")

    # ========================================================================
    # STEP 7: SAVE MODEL
    # ========================================================================

    print("\n" + "="*70)
    print("💾 SAVING MODEL")
    print("="*70)

    # Save LoRA adapters
    final_model_path = f"{config['OUTPUT_DIR']}/final_model"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"✅ LoRA adapters saved to: {final_model_path}")

    # Save merged 16-bit model
    merged_model_path = f"{config['OUTPUT_DIR']}/final_model_merged_16bit"
    model.save_pretrained_merged(
        merged_model_path,
        tokenizer,
        save_method="merged_16bit"
    )
    print(f"✅ Merged 16-bit model saved to: {merged_model_path}")

    # Commit volume changes
    volume.commit()

    print("\n" + "="*70)
    print("🎉 FINE-TUNING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Model saved to Modal volume at: {config['OUTPUT_DIR']}")
    print("You can now use the model for inference!")

    return {
        "status": "success",
        "model_path": final_model_path,
        "merged_model_path": merged_model_path,
    }


# ============================================================================
# LOCAL ENTRYPOINT
# ============================================================================

@app.local_entrypoint()
def main():
    """
    Main entry point that triggers the fine-tuning job.
    Run with: modal run train_modal.py
    """
    print("🚀 Launching fine-tuning job on Modal...")
    print("This will run on a cloud GPU (T4)")
    print("You can close this terminal - the job will continue in the cloud")
    print("")

    # Upload dataset to Modal
    print("📤 Uploading dataset to Modal...")

    # Run the fine-tuning
    result = finetune_model.remote()

    print("\n✅ Job completed!")
    print(f"Result: {result}")


# ============================================================================
# ALTERNATIVE: RUN IN BACKGROUND
# ============================================================================

@app.function(schedule=modal.Cron("0 0 * * *"))  # Optional: schedule daily
def scheduled_training():
    """
    Optional: Schedule training to run automatically.
    """
    finetune_model.remote()


if __name__ == "__main__":
    # You can also run this locally for testing
    print("💡 Tip: Run with 'modal run train_modal.py' to execute on Modal's cloud GPUs")
