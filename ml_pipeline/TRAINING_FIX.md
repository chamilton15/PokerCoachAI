# Quick Fix for Training

## The Problem
- `python3` command uses system Python 3.9 (architecture mismatch)
- NumPy is installed for conda Python 3.12
- PyTorch needs to be installed too

## Solution: Use Conda Python

Since you're in conda environment (base), use the conda Python directly:

### Option 1: Use `python` instead of `python3`

```bash
cd /Users/sethfgn/Desktop/DL_Poker_Project/Github_Repo/PokerCoachAI/ml_pipeline

# Check if conda Python works
python --version  # Should be 3.12

# Install PyTorch in conda environment
pip install torch

# Run training with conda Python
python train_model.py
```

### Option 2: Install PyTorch first, then train

```bash
# Make sure you're in conda environment
conda activate base

# Install PyTorch
pip install torch

# Verify it works
python -c "import torch; import numpy; print('✓ All good!')"

# Run training
python train_model.py
```

## Quick Check Commands

```bash
# Check which Python you're using
which python
python --version

# Check if numpy works
python -c "import numpy; print('NumPy works!')"

# Check if torch works (after installing)
python -c "import torch; print(f'PyTorch {torch.__version__} works!')"
```

## Expected Successful Run

Once PyTorch is installed, you should see:

```
================================================================================
TRAINING LSTM NEURAL NETWORK FOR POKER DECISION PREDICTION
================================================================================

Using device: cpu

Loading dataset from dataset/dataset.pkl...
...
```

No more "NumPy not available" error!

