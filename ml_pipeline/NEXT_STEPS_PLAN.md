# Next Steps: Neural Network Training & User Analysis

## 🎯 Goal
Create a neural network that:
1. **Trains** on winner decision data (learn optimal play patterns)
2. **Analyzes** user's hand history from `.phh`/`.phhs` files
3. **Identifies** weaknesses (where user deviates from optimal)
4. **Recommends** improvements based on patterns

---

## 📋 Step-by-Step Plan

### **STEP 1: Build & Train Neural Network**

#### 1.1 Architecture Design
**Input:**
- Current state features (11 features)
- Action history sequence (20 actions × 5 features)

**Architecture Options:**

**Option A: Simple Feedforward (Fastest to implement)**
```
Input Current State (11) → Dense(64) → Dense(32)
Input History (20×5) → Flatten → Dense(64) → Dense(32)
Concat → Dense(64) → Dense(32) → Output(3: fold/call/raise)
```

**Option B: LSTM for Sequence (Better for action history)**
```
Current State (11) → Dense(32)
History Sequence → LSTM(32) → Dense(32)
Concat → Dense(64) → Output(3)
```

**Option C: Transformer (Best performance, most complex)**
```
Current State → Embedding
History Sequence → Multi-head Attention → Dense
Concat → Dense → Output(3)
```

**Recommendation: Start with Option A, upgrade to B if needed**

#### 1.2 Training Pipeline
- Load dataset from `dataset/dataset.pkl`
- Build model architecture
- Train on train split
- Validate on val split
- Save trained model

**Output:** `models/poker_coach_model.pkl` or `.h5`

---

### **STEP 2: User Hand History Analysis Pipeline**

#### 2.1 Parse User's Hand History
- Load user's `.phh` or `.phhs` file
- Extract all their decision points (same as we did for winners)
- For each decision, extract:
  - Current game state
  - Action history up to that point

#### 2.2 Model Prediction
For each user decision:
1. Encode state + history (same encoding as training)
2. Run through trained model → Get probability distribution
3. Model outputs: `[P(fold), P(call), P(raise)]`
4. Optimal action = argmax of probabilities

#### 2.3 Compare User vs Optimal
```python
user_action = 'fold'  # What user actually did
optimal_action = 'call'  # What model recommends (highest prob)
confidence = 0.85  # Model's confidence

if user_action != optimal_action:
    if confidence > 0.70:
        # Flag as potential mistake
        mistake = {
            'hand_id': 123,
            'user_action': 'fold',
            'optimal_action': 'call',
            'confidence': 0.85,
            'state': {...},
            'severity': 'high' if confidence > 0.80 else 'medium'
        }
```

---

### **STEP 3: Weakness Identification**

#### 3.1 Pattern Detection
Group mistakes by:
1. **Mistake Type:**
   - Over-folding (folded when should call/raise)
   - Over-calling (called when should fold/raise)
   - Under-aggressive (should raise but called/folded)
   - Over-aggressive (raised when should fold/call)

2. **Situation:**
   - Preflop from early position
   - Preflop from late position
   - Postflop heads-up
   - Postflop multi-way
   - Facing raise
   - Facing 3bet

3. **Position:**
   - UTG mistakes
   - Button mistakes
   - etc.

#### 3.2 Statistical Analysis
Calculate:
- Mistake frequency by type
- Mistake frequency by position
- Estimated BB/100 cost (if available)
- Confidence-weighted severity

---

### **STEP 4: Recommendation Generation**

#### 4.1 Identify Top Weaknesses
Rank mistakes by:
- Frequency (most common)
- Confidence (highest model confidence)
- Estimated impact (BB cost)

#### 4.2 Generate Recommendations
For each top weakness pattern:

**Example 1: Over-folding to raises**
```
Pattern: Folded to raises 85% of time (25 instances)
Model says: Should defend ~45% of time
Recommendation: "You're over-folding to raises. A balanced strategy 
defends ~45% of hands in these spots. Over-folding makes you 
exploitable - opponents can profitably bluff against you with 
almost any two cards."
```

**Example 2: Playing too tight from Button**
```
Pattern: Only opened 18% from button (should be ~35-40%)
Model says: Should raise more from BTN
Recommendation: "You're playing too tight from the button. As the 
last to act, you can open wider (35-40% of hands). You're missing 
profitable stealing opportunities."
```

---

## 📁 File Structure

```
ml_pipeline/
├── train_model.py              # NEW: Train neural network
├── models/
│   └── poker_coach_model.pkl   # NEW: Trained model
├── analyze_player.py           # NEW: Analyze user's hand history
├── generate_report.py          # NEW: Create weakness report
└── reports/
    └── user_analysis.json      # NEW: Analysis output
```

---

## 🔧 Implementation Details

### **train_model.py**
```python
1. Load dataset from dataset/dataset.pkl
2. Convert lists to numpy arrays (if needed)
3. Build neural network (PyTorch/TensorFlow/Keras)
4. Train with:
   - Loss: Categorical cross-entropy
   - Optimizer: Adam
   - Metrics: Accuracy
5. Validate on val set
6. Save model
```

### **analyze_player.py**
```python
1. Load user's .phh/.phhs file
2. Extract decision points (reuse extract_decision_with_history)
3. For each decision:
   a. Encode state + history
   b. Model.predict() → get optimal action
   c. Compare to user's actual action
   d. Flag if mismatch (high confidence)
4. Group mistakes by pattern
5. Return mistake analysis
```

### **generate_report.py**
```python
1. Load mistake analysis
2. Identify top patterns:
   - Most frequent mistake types
   - Position-specific weaknesses
   - Situation-specific weaknesses
3. Generate recommendations:
   - Top 3-5 weaknesses
   - Actionable advice
   - Estimated impact
4. Create readable report (JSON + text summary)
```

---

## 🎯 Expected Workflow

```bash
# 1. Train the model (one time)
python train_model.py

# 2. Analyze a user's session
python analyze_player.py --input user_session.phhs --player "PlayerName"

# 3. Generate report
python generate_report.py --analysis user_analysis.json

# Output: reports/user_report.txt + user_report.json
```

---

## 📊 Expected Output Example

```
PLAYER ANALYSIS REPORT
======================

Session: user_session.phhs
Hands analyzed: 150
Decision points: 487

MISTAKE SUMMARY:
Total mistakes: 43 (8.8% of decisions)
High confidence mistakes: 28 (5.8%)

TOP WEAKNESSES:

1. Over-folding to raises (HIGH SEVERITY)
   Frequency: 18 instances (42% of mistakes)
   Situation: Facing raise from late position
   Model recommendation: Defend 45% of range
   Your defense rate: 15%
   Impact: ~5 BB/100 (estimated)

   Recommendation: "You're folding too often to raises. In late 
   position vs late position, you should defend ~45% of your 
   opening range. Your current 15% defense rate is too tight."

2. Playing too tight from Button (MEDIUM SEVERITY)
   Frequency: 12 instances (28% of mistakes)
   Situation: Preflop, facing no action
   Model recommendation: Open 35-40% from BTN
   Your open rate: 18%
   Impact: ~3 BB/100 (estimated)

3. Under-aggressive on flop (MEDIUM SEVERITY)
   ...
```

---

## 🚀 Quick Start Priority

**Phase 1 (MVP):**
1. ✅ Train simple feedforward model
2. ✅ Basic player analysis (compare actions)
3. ✅ Simple mistake counting
4. ✅ Top 3 weakness identification

**Phase 2 (Enhanced):**
1. LSTM/Transformer architecture
2. Confidence-weighted analysis
3. Estimated BB cost calculation
4. Detailed recommendations

**Phase 3 (Advanced):**
1. Position-specific analysis
2. Situation-specific analysis
3. Session comparison (track improvement)
4. Interactive feedback

---

## 💡 Key Design Decisions

1. **Model predicts action probabilities** → Can show "confidence"
2. **Compare user action vs optimal** → Flag mismatches
3. **Group by patterns** → Identify systemic weaknesses
4. **Prioritize by frequency + confidence** → Focus on real issues
5. **Actionable recommendations** → Tell user HOW to improve

---

## 🔄 Integration with Existing Code

- **Reuse:** `extract_decision_with_history()` from extract_training_data.py
- **Reuse:** Feature encoding from prepare_training_dataset.py
- **New:** Model training
- **New:** Player analysis pipeline
- **New:** Report generation

---

Ready to start implementing? Recommend starting with `train_model.py` using a simple feedforward network first!

