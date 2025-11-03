# Neural Network Training Guide

## 🎯 What We're Building

An LSTM neural network that learns optimal poker decisions from winning players' data.

**Architecture:** Option B - LSTM for sequence processing
- **Input 1:** Current game state (11 features)
- **Input 2:** Action history sequence (20 actions × 5 features)
- **Output:** Probability distribution [P(fold), P(call), P(raise)]
- **Architecture:** Current state → Dense layers, History → LSTM → Combine → Output

---

## 📋 Step-by-Step Instructions

### **STEP 1: Install Dependencies**

You need PyTorch and NumPy:

```bash
# Install PyTorch (choose appropriate version for your system)
pip3 install torch numpy

# Or if using conda:
conda install pytorch numpy -c pytorch
```

**Verify installation:**
```bash
python3 -c "import torch; import numpy; print(f'PyTorch: {torch.__version__}')"
```

---

### **STEP 2: Train the Model**

```bash
cd /Users/sethfgn/Desktop/DL_Poker_Project/Github_Repo/PokerCoachAI/ml_pipeline
python3 train_model.py
```

**What happens:**
1. Loads dataset from `dataset/dataset.pkl`
2. Creates train/val/test splits (already done)
3. Builds LSTM model architecture
4. Trains for 50 epochs
5. Validates after each epoch
6. Saves best model based on validation accuracy
7. Tests on held-out test set
8. Reports baseline comparisons

**Expected output:**
```
================================================================================
TRAINING LSTM NEURAL NETWORK FOR POKER DECISION PREDICTION
================================================================================

Using device: cpu  (or cuda if GPU available)

Loading dataset from dataset/dataset.pkl...
Dataset dimensions:
  Current features: 11
  History sequence: 20 actions × 5 features
  Number of classes: 3

Data splits:
  Train: 37,316 examples
  Validation: 4,664 examples
  Test: 4,665 examples

Initializing LSTM model...
  Total parameters: ~45,000
  Trainable parameters: ~45,000

================================================================================
TRAINING
================================================================================
Epoch [1/50]
  Train Loss: 0.8234 | Train Acc: 58.23%
  Val Acc: 59.45%
  ✓ Saved best model (val_acc: 59.45%)

Epoch [5/50]
  Train Loss: 0.7123 | Train Acc: 65.12%
  Val Acc: 64.89%

...

================================================================================
FINAL EVALUATION
================================================================================

Test Accuracy: 67.23%
Validation Accuracy: 66.45%

Baseline Comparisons:
  Random baseline (1/3): 33.33%
  Majority class baseline: 47.89% (class 1 - call)
  Model improvement over random: +33.90%
  Model improvement over majority: +19.34%

Per-class accuracy on test set:
  fold       :  72.34% (842/1164)
  call       :  65.12% (1523/2339)
  raise      :  61.45% (715/1162)
```

---

### **STEP 3: Understanding the Results**

#### **Baseline Accuracy Targets:**

1. **Random Baseline:** 33.33% (guessing randomly)
   - **Target:** Beat by at least +20% → Aim for >53%

2. **Majority Class Baseline:** ~48% (always predict "call")
   - **Target:** Beat by at least +15% → Aim for >63%

3. **Good Performance:** 
   - **Target:** >65% overall accuracy
   - **Why:** Poker has inherent randomness, 65%+ means model learned real patterns

4. **Excellent Performance:**
   - **Target:** >70% overall accuracy
   - This would indicate strong learning from winner patterns

#### **What the Metrics Mean:**

- **Train Accuracy:** How well model fits training data
- **Validation Accuracy:** How well model generalizes (most important!)
- **Test Accuracy:** Final performance on unseen data
- **Per-class Accuracy:** How well model predicts each action type

**Key Insight:** If validation accuracy is much lower than train accuracy → overfitting (reduce complexity or add regularization)

---

### **STEP 4: Model Output Files**

After training, you'll get:

```
models/
├── poker_coach_model.pt         # The trained model (PyTorch format)
└── training_history.json        # Training curves and metrics
```

**`poker_coach_model.pt` contains:**
- Model weights (learned parameters)
- Model architecture details
- Metadata (feature dimensions, etc.)
- Best validation accuracy

---

### **STEP 5: How Model Will Be Used for Analysis**

Once trained, here's how it analyzes new player `.phh` files:

```
1. User provides: user_session.phhs (their hand history)

2. Extract decisions (same as we did for winners):
   - Parse .phhs file
   - Extract each decision point
   - Encode state + history

3. Model prediction for each decision:
   - Input: Game state + action history
   - Model outputs: [P(fold)=0.10, P(call)=0.75, P(raise)=0.15]
   - Optimal action = "call" (highest probability)

4. Compare to user's actual action:
   - User actually did: "fold"
   - Model recommends: "call"
   - Confidence: 75%
   - → Flag as potential mistake (if confidence > threshold)

5. Aggregate mistakes:
   - Group by type (over-folding, under-aggressive, etc.)
   - Rank by frequency
   - Generate recommendations
```

---

## 🔧 Configuration Options

You can modify training parameters in `train_model.py`:

```python
# In main() function:
epochs = 50              # Number of training epochs (increase for better results)
batch_size = 64          # Batch size (32-128 recommended)
learning_rate = 0.001    # Learning rate (0.0001-0.01 range)
```

**Recommendations:**
- **More epochs** = Better accuracy (but slower, risk of overfitting)
- **Larger batch size** = Faster training (requires more memory)
- **Lower learning rate** = More stable training (but slower)

---

## 🎯 Success Criteria

**Minimum Viable:**
- ✅ Test accuracy > 50% (beats random)
- ✅ Validation accuracy > 55% (beats majority class)

**Good Performance:**
- ✅ Test accuracy > 65%
- ✅ All per-class accuracies > 55%
- ✅ Validation accuracy tracks closely with train accuracy

**Excellent Performance:**
- ✅ Test accuracy > 70%
- ✅ Balanced per-class accuracies (no class < 60%)
- ✅ Stable training (no overfitting)

---

## 🐛 Troubleshooting

### **Issue: "PyTorch not installed"**
```bash
pip3 install torch
```

### **Issue: Low accuracy (<50%)**
- Increase epochs (try 100)
- Decrease learning rate (try 0.0005)
- Check if dataset is valid

### **Issue: Overfitting (val acc << train acc)**
- Model too complex → Already using dropout, should be fine
- Train for fewer epochs
- Increase dropout rates in model

### **Issue: CUDA out of memory**
- Reduce batch_size to 32 or 16
- Training will be slower but use less memory

---

## 📊 Expected Training Time

- **CPU:** ~15-30 minutes for 50 epochs
- **GPU:** ~2-5 minutes for 50 epochs

You'll see progress every 5 epochs, so you can stop early (Ctrl+C) if satisfied.

---

## 🔄 Next Steps After Training

Once you have a trained model:

1. **Analyze player hand history:**
   - Create `analyze_player.py` (next step)
   - Uses trained model to analyze `.phh` files

2. **Generate weakness reports:**
   - Create `generate_report.py` (after analysis)
   - Identifies patterns and recommendations

---

## 📝 Quick Reference Commands

```bash
# Train model
python3 train_model.py

# Check if model was created
ls -lh models/poker_coach_model.pt

# View training history
cat models/training_history.json
```

---

Ready to train! Run `python3 train_model.py` and let it train. The model will automatically save the best version based on validation accuracy.

