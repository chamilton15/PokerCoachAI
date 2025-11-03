# Neural Network Approach - Implementation Guide

## Overview

This pipeline converts the rule-based system to a **neural network that learns from winning players' actual behavior**, including full action history context.

---

## Architecture Decision

### Why Neural Network Instead of Rules?

**Old Approach (Rule-Based):**
- Hardcoded GTO ranges
- Simple if-then logic
- ~80% accuracy, not learnable

**New Approach (Neural Network):**
- Learns from real winning players
- Discovers patterns automatically
- Improves with more data
- Can handle complex contexts

---

## Training Data Approach

### Supervised Learning from Winners (Selected)

**Why this over RL:**
- Faster convergence (clear labels)
- Lower variance (learns consistent patterns)
- More data-efficient
- Directly learns "what winners do"

### Data Source: Top 20% Winners from handhq

**Rationale:**
- Large dataset (21M+ hands)
- Diverse playing conditions
- Real online poker data
- Can identify consistent winners

**Selection Criteria:**
- Minimum 20 hands (reliable sample)
- Top 20% by BB/hand (proven profitability)
- All players meeting threshold

---

## Action History - The Key Innovation

### Why Include Full History?

Poker decisions depend heavily on context:

```
Bad Context (No History):
"Hero has 77, position BTN, pot $2.25"
→ Model: "Raise!" (might be wrong)

Good Context (With History):
"Hero has 77, position BTN, pot $2.25
 Previous actions:
  - UTG raised to $1.50 (tight player!)
  - MP called (loose player)
  - CO folded"
→ Model: "Call!" (more informed decision)
```

### What History Includes

For each decision point, the model sees:
1. All previous actions in the hand
2. Who acted (player positions)
3. What they did (fold/call/raise)
4. How much they bet (if raised)
5. Which street (preflop/flop/turn/river)
6. Pot size after each action

**This gives complete context!**

---

## Feature Engineering

### Current State (11 features):
- Hand strength (1-10 score)
- Hand type (pair/suited/offsuit)
- Position (UTG/BTN/etc.)
- Street (preflop/flop/turn/river)
- Pot size (normalized)
- Stack size (normalized)
- Number of opponents
- Facing action (none/raise/3bet)
- History length
- Raises in history
- Calls in history

### Action History (Sequence):
- Fixed-length: 20 timesteps (max actions in a hand)
- Each timestep: [player_idx, action, amount, street, pot]
- Padded/truncated as needed
- Normalized for neural network

### Labels:
- 3 classes: fold (0), call (1), raise (2)
- One-hot encoded for classification

---

## Network Architecture (Next Step)

### Recommended: Hybrid Model

```
Input 1: Current Features (11 dim) → Feedforward Branch
Input 2: Action History (20×5) → LSTM/Transformer Branch

Concatenate both branches → Final Dense Layers → Output (3 classes)
```

**Why Hybrid:**
- Feedforward: Learns from current state
- LSTM: Learns from action sequences
- Combined: Best of both worlds

---

## Training Strategy

### Phase 1: Pre-training
- Train on all winners' data
- Learn general winning patterns
- Establish baseline performance

### Phase 2: Fine-tuning (Optional)
- Fine-tune on specific player types
- Adapt to different styles
- Specialize for tournament vs cash

### Phase 3: Evaluation
- Test on held-out winners
- Test on Hero (does it identify leaks?)
- Compare to baseline rule-based system

---

## Expected Improvements

### Over Rule-Based System:

1. **Context Awareness:**
   - Rule-based: "77 from BTN → Raise"
   - Neural: "77 from BTN, but UTG raised, so maybe fold/call"

2. **Pattern Discovery:**
   - Rule-based: Fixed ranges
   - Neural: Discovers when to deviate

3. **Adaptability:**
   - Rule-based: Hand-coded
   - Neural: Improves with more data

4. **Complexity:**
   - Rule-based: Can't handle all situations
   - Neural: Learns complex dependencies

---

## Challenges & Solutions

### Challenge 1: Class Imbalance
**Problem:** Folds > Calls > Raises (skewed distribution)

**Solutions:**
- Weight classes inversely proportional to frequency
- Oversample minority classes (raises)
- Use focal loss for hard examples

### Challenge 2: Sequence Length Variation
**Problem:** Hands have different numbers of actions

**Solutions:**
- Fixed-length sequences (20 max)
- Padding with zeros for short sequences
- Truncate long sequences (keep last 20)

### Challenge 3: Overfitting
**Problem:** Model memorizes specific players

**Solutions:**
- Regularization (dropout, L2)
- Train on diverse winners
- Early stopping on validation set
- Data augmentation (shuffle action order variations)

---

## Data Flow Summary

```
handhq/.phhs files
    ↓
Step 1: extract_winners.py
    ├─ Parse all files
    ├─ Calculate BB/hand per player
    ├─ Select top 20% winners
    └─ Output: winners_list.json
    ↓
Step 2: extract_training_data.py
    ├─ For each winner's hand:
    │  ├─ Extract each decision point
    │  ├─ Include full action history
    │  └─ Record label (what winner did)
    └─ Output: training_data.json
    ↓
Step 3: prepare_training_dataset.py
    ├─ Encode all features numerically
    ├─ Create fixed-length sequences
    ├─ Prepare labels (one-hot)
    ├─ Split train/val/test
    └─ Output: dataset/dataset.pkl
    ↓
Step 4: train_model.py (TO BE IMPLEMENTED)
    ├─ Load prepared dataset
    ├─ Build neural network
    ├─ Train on training set
    ├─ Validate on validation set
    └─ Save trained model
    ↓
Step 5: analyze_player.py (TO BE IMPLEMENTED)
    ├─ Load trained model
    ├─ Extract player's decisions
    ├─ Predict optimal actions
    ├─ Compare to actual actions
    └─ Generate recommendations
```

---

## Key Files

### Data Extraction:
- `extract_winners.py` - Step 1: Identify winners
- `extract_training_data.py` - Step 2: Extract with history
- `prepare_training_dataset.py` - Step 3: Encode and prepare

### To Be Implemented:
- `train_model.py` - Neural network training
- `analyze_with_model.py` - Use trained model for analysis

---

## Next Implementation Steps

1. ✅ Data extraction pipeline (DONE)
2. ✅ Feature encoding (DONE)
3. ⏳ Neural network architecture
4. ⏳ Training loop
5. ⏳ Evaluation metrics
6. ⏳ Player analysis using trained model

---

**The foundation is ready! Next: Build and train the neural network.**


