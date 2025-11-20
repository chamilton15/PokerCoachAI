# Poker Coach AI - ML Pipeline

Complete guide to the machine learning pipeline for training a neural network that learns optimal poker strategies from winning players' hand histories.

---

## 📋 Overview

This pipeline trains an LSTM neural network to predict optimal poker actions (fold/call/raise) by learning from the decision patterns of top-performing players. The model can then analyze any player's hand history and provide personalized coaching feedback.

**Pipeline Flow:**
1. **Identify Winners** → Extract top 20% profitable players
2. **Extract Training Data** → Get decision points with full game context
3. **Prepare Dataset** → Encode features and split into train/val/test
4. **Train Model** → Train LSTM neural network
5. **Analyze Players** → (Future) Use trained model to analyze user sessions

---

## 🚀 Quick Start

```bash
# 1. Identify winning players
python extract_winners.py

# 2. Extract training data from winners
python extract_training_data.py

# 3. Prepare dataset for training
python prepare_training_dataset.py

# 4. Train the neural network
python train_model.py
```

---

## 📁 Directory Structure

```
ml_pipeline/
├── extract_winners.py          # Step 1: Identify top 20% winners
├── extract_training_data.py    # Step 2: Extract decision points
├── prepare_training_dataset.py # Step 3: Encode and split data
├── train_model.py              # Step 4: Train LSTM model
│
├── winners_list.json           # Output: List of winning players
├── training_data.json          # Output: Raw training examples
├── dataset/
│   ├── dataset.pkl             # Output: Encoded dataset
│   └── dataset_metadata.json   # Output: Dataset info
└── models/
    ├── poker_coach_model.pt    # Output: Trained model
    └── training_history.json   # Output: Training metrics
```

---

## 🔧 Prerequisites

### Python Environment
- Python 3.9+ (or Python 3.12 with conda)
- Required packages:
  ```bash
  pip install torch numpy
  ```

### Data Requirements
- Poker hand history files (`.phhs` format) in:
  ```
  ../../Poker_Data_Set/data/handhq/ABS-2009-07-01_2009-07-23_50NLH_OBFU/0.5/
  ```

### System Requirements
- **CPU:** 4+ cores recommended for parallel processing
- **RAM:** 8GB+ recommended
- **Disk:** ~5GB free space for datasets and models

---

## 📖 Step-by-Step Guide

### **Step 1: Identify Winning Players**

**Script:** `extract_winners.py`

**What it does:**
- Scans all `.phhs` files in the handhq dataset
- Calculates BB/hand profitability for each player
- Identifies top 20% of players (minimum 20 hands played)
- Saves list of winners with statistics

**How to run:**
```bash
cd /Users/sethfgn/Desktop/DL_Poker_Project/Github_Repo/PokerCoachAI/ml_pipeline
python extract_winners.py
```

**Expected output:**
- `winners_list.json` - List of winning players with stats
- Progress updates every 50 files
- Final summary with top winners

**Example output:**
```
Processing files...
  Processed 50/200 files...
  Processed 100/200 files...
  
Top 10 Winners:
  1. Player1: 179.23 BB/hand (1,234 hands)
  2. Player2: 156.45 BB/hand (892 hands)
  ...
```

**Time:** ~10-30 minutes (depending on dataset size)

---

### **Step 2: Extract Training Data**

**Script:** `extract_training_data.py`

**What it does:**
- For each winning player, extracts every decision point they made
- Captures full game state at each decision:
  - Current state: hand strength, position, pot size, stack, etc.
  - **Action history: ALL players' actions** (not just winner's!)
  - Label: What the winner actually did (fold/call/raise)

**How to run:**
```bash
python extract_training_data.py
```

**Expected output:**
- `training_data.json` - All training examples with full context
- Progress updates every 50 files
- Summary of extracted examples

**Example structure:**
```json
{
  "training_data": [
    {
      "hand_id": 123,
      "state": {
        "current_state": {
          "hole_cards": ["Ah", "Kh"],
          "position": "BTN",
          "pot_size": 2.25,
          ...
        },
        "action_history": [
          {"player_idx": 0, "action_type": "raise", ...},
          {"player_idx": 1, "action_type": "call", ...},
          ...
        ]
      },
      "label": "raise"
    }
  ]
}
```

**Time:** ~15-45 minutes (depends on number of winners)

---

### **Step 3: Prepare Training Dataset**

**Script:** `prepare_training_dataset.py`

**What it does:**
- Encodes all features numerically:
  - Hand strength → normalized score
  - Position → integer encoding
  - Action history → fixed-length sequences (20 actions × 5 features)
  - Labels → one-hot encoding
- Splits data: 80% train, 10% validation, 10% test
- Saves as pickle file for efficient loading

**How to run:**
```bash
python prepare_training_dataset.py
```

**Expected output:**
- `dataset/dataset.pkl` - Encoded dataset ready for training
- `dataset/dataset_metadata.json` - Dataset dimensions and info

**Example output:**
```
Dataset shape:
  Current features: 112,431 examples, 11 features
  History sequences: 112,431 examples, 20 actions, 5 features per action
  Labels: 112,431 examples

Label distribution:
  call      : 53,846 (52.9%)
  raise     : 27,164 (26.7%)
  fold      : 20,552 (20.2%)

Splitting dataset (80/10/10)...
  Train: 89,945 examples
  Val: 11,243 examples
  Test: 11,243 examples
```

**Time:** ~2-5 minutes

---

### **Step 4: Train Neural Network**

**Script:** `train_model.py`

**What it does:**
- Loads prepared dataset
- Builds LSTM model architecture:
  - Current state encoder (Dense layers)
  - Action history encoder (LSTM layers)
  - Combined classifier
- Trains for 50 epochs with validation
- Saves best model based on validation accuracy
- Reports final test accuracy and baselines

**How to run:**
```bash
# Make sure you're using conda Python (not system Python 3.9)
python train_model.py

# Or use conda Python directly:
/Users/sethfgn/miniforge3/bin/python train_model.py
```

**Expected output:**
```
Initializing LSTM model...
  Total parameters: 73,763
  Trainable parameters: 73,763

Class distribution:
  fold      : 2,147 (28.7%) - weight: 1.162
  call      : 3,718 (49.7%) - weight: 0.672
  raise     : 1,613 (21.6%) - weight: 1.548

Epoch [1/50]
  Train Loss: 0.8234 | Train Acc: 58.23%
  Val Acc: 59.45%
  ✓ Saved best model

Epoch [5/50]
  Train Loss: 0.7123 | Train Acc: 65.12%
  Val Acc: 64.89%
  Sample predictions: fold=0.312, call=0.423, raise=0.265

...

FINAL EVALUATION
Test Accuracy: 67.23%
Validation Accuracy: 66.45%

Baseline Comparisons:
  Random baseline (1/3): 33.33%
  Majority class baseline: 51.00%
  Model improvement over random: +33.90%
  Model improvement over majority: +19.34%
```

**Output files:**
- `models/poker_coach_model.pt` - Trained model weights
- `models/training_history.json` - Training curves and metrics

**Time:** ~15-30 minutes (CPU) or ~2-5 minutes (GPU)

---

## 🎯 Model Architecture

### Architecture Overview
- **Input 1:** Current state features (11 dimensions)
- **Input 2:** Action history sequence (20 actions × 5 features)
- **Architecture:** Two-path design
  - Current state → Dense(32) → Dense(32)
  - History → LSTM(64, 2 layers) → Extract last hidden state
  - Combined → Dense(128) → Dense(64) → Output(3 classes)

### Why LSTM?
- Action history is sequential - order matters!
- LSTM captures temporal dependencies (e.g., "UTG raised, then CO called")
- Can learn patterns like "after 2 raises, fold weak hands"

### Key Design Decisions
- **Class weights:** Handle imbalance (call is 49.7% vs fold 28.8%)
- **Gradient clipping:** Prevents exploding gradients
- **Dropout:** Reduces overfitting (0.2-0.1)
- **Learning rate:** 0.0001 for stability
- **Batch size:** 32 for stable gradients

---

## 📊 Expected Results

### Good Performance Metrics
- **Test accuracy:** >65% (beats random 33% and majority 51%)
- **Per-class accuracy:** All classes >55%
- **Validation tracks train:** No overfitting

### Success Indicators
- Model predicts all 3 classes (not just one)
- Validation accuracy improves over epochs
- Sample predictions show balanced probabilities

---

## 🔍 Troubleshooting

### Issue: "NumPy not available"
**Solution:** Use conda Python instead of system Python
```bash
conda activate base
python train_model.py
```

### Issue: "Loss is NaN"
**Possible causes:**
- Class weights too extreme → Already fixed with capping
- Invalid data → Check for NaN/Inf in dataset
- Learning rate too high → Already reduced to 0.0001

**Solution:** The code now includes:
- NaN detection and replacement
- Gradient clipping
- Class weight capping
- Better weight initialization

### Issue: "Model only predicts one class"
**Possible causes:**
- Class imbalance → Fixed with class weights
- Poor initialization → Fixed with Xavier/He init
- Data issues → Check action_history includes all players

**Solution:** Re-run extraction with fixed `extract_training_data.py` (includes all players' actions)

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in `train_model.py`:
```python
batch_size = 16  # Instead of 32
```

### Issue: "Training is slow"
**Solution:**
- Use GPU if available (automatic)
- Reduce batch size (faster per epoch)
- Reduce epochs if accuracy plateaus

---

## 📈 Understanding the Outputs

### `winners_list.json`
```json
{
  "winners": [
    {
      "player": "PlayerName",
      "total_hands": 1234,
      "bb_per_hand": 179.23,
      "total_bb": 221,050
    }
  ]
}
```

### `training_data.json`
Raw training examples with:
- Game state at decision point
- Full action history (all players)
- Winner's actual action (label)

### `dataset/dataset.pkl`
PyTorch-ready dataset with:
- Encoded features (numpy arrays)
- Train/val/test splits
- Metadata (dimensions, label maps)

### `models/poker_coach_model.pt`
Trained model containing:
- Model weights
- Architecture info
- Best validation accuracy
- Training metadata

### `models/training_history.json`
Training metrics:
- Loss curves
- Accuracy curves
- Per-class accuracy
- Baseline comparisons

---

## 🔄 Re-running the Pipeline

### Full Pipeline (from scratch)
```bash
# 1. Identify winners
python extract_winners.py

# 2. Extract training data
python extract_training_data.py

# 3. Prepare dataset
python prepare_training_dataset.py

# 4. Train model
python train_model.py
```

### Just Retrain Model (if data already prepared)
```bash
python train_model.py
```

### Update Training Data (if extraction changed)
```bash
python extract_training_data.py
python prepare_training_dataset.py
python train_model.py
```

---

## 🎯 Next Steps (Future)

### 1. Analyze Player Sessions
**Script:** `analyze_player.py` (to be created)
- Load user's `.phh` file
- Extract their decision points
- Compare to model predictions
- Identify mistakes

### 2. Generate Recommendations
**Script:** `generate_report.py` (to be created)
- Aggregate mistakes by pattern
- Rank by frequency/confidence
- Generate personalized feedback
- Optionally use LLM for natural language

### 3. Improve Model
- Increase training data (more winners)
- Tune hyperparameters
- Try different architectures (Transformer)
- Add more features (postflop cards, etc.)

---

## 📝 Key Files Explained

| File | Purpose | When to Run |
|------|---------|-------------|
| `extract_winners.py` | Find top 20% players | Once (or when data changes) |
| `extract_training_data.py` | Extract decision points | After winners identified |
| `prepare_training_dataset.py` | Encode and split data | After training data extracted |
| `train_model.py` | Train neural network | After dataset prepared |

---

## 🐛 Common Issues & Fixes

### Python Version Mismatch
**Problem:** System Python 3.9 vs Conda Python 3.12
**Fix:** Always use conda Python:
```bash
conda activate base
python train_model.py
```

### Architecture Mismatch (ARM64 vs x86_64)
**Problem:** NumPy/PyTorch compiled for wrong architecture
**Fix:** Use conda Python which has correct packages

### Out of Memory
**Problem:** Dataset too large for RAM
**Fix:** Reduce batch size or use data streaming

### Low Accuracy
**Problem:** Model not learning
**Fix:** 
- Check if action_history includes all players
- Verify class weights are reasonable
- Check for NaN in data
- Try different learning rate

---

## 📚 Additional Resources

- **Training Guide:** `TRAINING_GUIDE.md` - Detailed training instructions
- **Architecture Explanation:** `ML_APPROACH.md` - Why we chose this approach
- **Next Steps:** `NEXT_STEPS_PLAN.md` - Future development plans
- **Model Accuracy:** `MODEL_ACCURACY_EXPLANATION.md` - What accuracy means

---

## ✅ Quick Checklist

Before running the pipeline:
- [ ] Python 3.12 (conda) installed
- [ ] PyTorch installed: `pip install torch`
- [ ] Dataset files in correct location
- [ ] Enough disk space (~5GB)
- [ ] Terminal in `ml_pipeline/` directory

After each step:
- [ ] Check output file was created
- [ ] Verify output looks reasonable
- [ ] Check for error messages
- [ ] Note any warnings

---

## 🎓 Understanding the Pipeline

### Why This Approach?
1. **Supervised Learning:** Learn from winners' actual decisions
2. **Full Context:** Include all players' actions (not just winner's)
3. **Sequential Modeling:** LSTM captures temporal patterns
4. **Scalable:** Can add more data/features easily

### What Makes It Work?
- **Large dataset:** 21M+ hands, top 20% winners
- **Rich features:** Game state + full action history
- **Proper validation:** Train/val/test splits prevent overfitting
- **Class balancing:** Weights handle label imbalance

---

**Ready to train?** Start with Step 1: `python extract_winners.py`
