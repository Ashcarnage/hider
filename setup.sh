#!/bin/bash

# 🚀 HIDER Fine-tuning Setup Script
# This script will help you set up and run the Modal fine-tuning pipeline

set -e  # Exit on error

echo "=============================================="
echo "🎮 HIDER Agent Fine-tuning Setup"
echo "=============================================="
echo ""

# Step 1: Check Python version
echo "📋 Step 1: Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "   Python version: $python_version"
echo ""

# Step 2: Install Modal
echo "📦 Step 2: Installing Modal..."
echo "   Running: pip install modal"
pip install modal
echo "   ✅ Modal installed!"
echo ""

# Step 3: Setup Modal account
echo "🔑 Step 3: Setting up Modal account..."
echo ""
echo "   IMPORTANT: This will open your browser to authenticate."
echo "   Steps:"
echo "   1. Create a Modal account (or login if you have one)"
echo "   2. You'll get $30 in FREE credits!"
echo "   3. Authorize the CLI"
echo ""
read -p "   Press ENTER to continue..." dummy
echo ""

modal setup

echo ""
echo "   ✅ Modal setup complete!"
echo ""

# Step 4: Verify setup
echo "✅ Step 4: Verifying setup..."
if modal token show > /dev/null 2>&1; then
    echo "   ✅ Modal is authenticated!"
else
    echo "   ❌ Modal authentication failed. Please run 'modal setup' manually."
    exit 1
fi
echo ""

# Step 5: Check dataset
echo "📊 Step 5: Checking dataset..."
if [ -f "hider_raw.jsonl" ]; then
    line_count=$(wc -l < hider_raw.jsonl)
    echo "   ✅ Dataset found: $line_count examples"
else
    echo "   ❌ Dataset not found! Please ensure hider_raw.jsonl is in this directory."
    exit 1
fi
echo ""

echo "=============================================="
echo "🎉 Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. 🏋️  Train the model (takes ~20-30 mins):"
echo "   modal run train_modal.py"
echo ""
echo "2. 🧪 Test the model:"
echo "   modal run test_model.py"
echo ""
echo "3. 📊 Monitor your jobs:"
echo "   Visit: https://modal.com/dashboard"
echo ""
echo "💡 Tips:"
echo "   - You have $30 in free credits (~$0.40 per training run)"
echo "   - You can close terminal during training - it runs in the cloud!"
echo "   - Check SETUP.md for detailed documentation"
echo ""
