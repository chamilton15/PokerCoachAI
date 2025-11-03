# How to Run Training - Quick Guide

## The Issue
Your system has Python 3.9 (x86_64) but numpy is installed for Python 3.12 (conda). You need to use conda Python.

## Solution: Use Conda Python Directly

### Step 1: Find Your Conda Python

```bash
# Find conda Python path (usually in miniforge3 or anaconda3)
/Users/sethfgn/miniforge3/bin/python --version

# Should show: Python 3.12.x
```

### Step 2: Install PyTorch for Conda Python

```bash
# Use conda Python directly to install
/Users/sethfgn/miniforge3/bin/python -m pip install torch

# Or if conda is activated:
pip install torch
```

### Step 3: Run Training with Conda Python

```bash
cd /Users/sethfgn/Desktop/DL_Poker_Project/Github_Repo/PokerCoachAI/ml_pipeline

# Option A: Use full path to conda Python
/Users/sethfgn/miniforge3/bin/python train_model.py

# Option B: Activate conda first, then use python
conda activate base
python train_model.py
```

## Quick Check Before Running

```bash
# 1. Check conda Python works
/Users/sethfgn/miniforge3/bin/python --version

# 2. Check numpy works
/Users/sethfgn/miniforge3/bin/python -c "import numpy; print('✓ NumPy works')"

# 3. Check torch works (after installing)
/Users/sethfgn/miniforge3/bin/python -c "import torch; print(f'✓ PyTorch {torch.__version__} works')"
```

## Alternative: Create a Shebang Script

Or modify `train_model.py` to use conda Python directly.

