# Training vs Testing - How It Works

## 🔄 The Three Phases

### 1. **TRAINING Phase** (Epochs 1-50)
**When:** During each epoch  
**What happens:**
- Model sees **training data** (37,316 examples)
- Makes predictions
- Compares to actual winner actions
- Adjusts weights (learns from mistakes)
- **Model gets better with each batch**

**Example:**
```
Epoch 1:
  Batch 1: Model predicts "call" → Actual was "raise" → Wrong! → Adjust weights
  Batch 2: Model predicts "fold" → Actual was "fold" → Correct! → Slight adjustment
  ...
  Result: Model accuracy improves from 33% → 58%
```

**Key Point:** Model **updates its weights** during training

---

### 2. **VALIDATION Phase** (After each epoch)
**When:** After training phase in each epoch  
**What happens:**
- Model sees **validation data** (4,664 examples) - **never used for training**
- Makes predictions (no weight updates!)
- Calculates accuracy
- Compares to previous best
- **Saves model if this is the best so far**

**Example:**
```
Epoch 5:
  Training Acc: 65.12% (on training data)
  Validation Acc: 64.89% (on validation data) ← Checks generalization
  Best so far? Yes! → Save model ✓
```

**Key Point:** Validation checks if model **generalizes** to unseen data (not just memorizing)

**Why Important:**
- If train acc >> val acc → Model is **overfitting** (memorizing, not learning patterns)
- If train acc ≈ val acc → Model is **generalizing well**

---

### 3. **TESTING Phase** (After all training)
**When:** After all 50 epochs complete  
**What happens:**
- Model sees **test data** (4,665 examples) - **never seen before**
- Makes final predictions
- Calculates **final accuracy**
- Reports baseline comparisons
- **No weight updates** - just evaluation!

**Example:**
```
After Epoch 50:
  Final Test Accuracy: 67.23%
  Validation Accuracy: 66.45%
  
  Baselines:
    Random: 33.33%
    Majority class: 47.89%
    Model beats random by: +33.90%
    Model beats majority by: +19.34%
```

**Key Point:** Test set is the **final, unbiased evaluation** of model performance

---

## 📊 Visual Timeline

```
START
  ↓
[Epoch 1]
  ├─ Training:  37,316 examples → Learn → Train Acc: 58%
  ├─ Validation: 4,664 examples → Check → Val Acc: 59% → Save! ✓
  └─ Test: Not used yet
  ↓
[Epoch 2]
  ├─ Training:  37,316 examples → Learn → Train Acc: 62%
  ├─ Validation: 4,664 examples → Check → Val Acc: 61% → Don't save
  └─ Test: Not used yet
  ↓
...
[Epoch 50]
  ├─ Training:  37,316 examples → Learn → Train Acc: 72%
  ├─ Validation: 4,664 examples → Check → Val Acc: 66% → Already saved at Epoch 45
  └─ Test: Not used yet
  ↓
[Final Test] ← Only runs once!
  ├─ Test: 4,665 examples → Evaluate → Test Acc: 67.23%
  └─ Report final results
```

---

## 🎯 Key Differences

| Aspect | Training | Validation | Testing |
|--------|----------|------------|---------|
| **When** | Every epoch | Every epoch | After all epochs |
| **Data** | 37,316 examples | 4,664 examples | 4,665 examples |
| **Purpose** | Learn patterns | Check generalization | Final evaluation |
| **Weight Updates** | ✅ YES | ❌ NO | ❌ NO |
| **Used For** | Teaching model | Picking best model | Reporting results |
| **Frequency** | 50 times | 50 times | 1 time |

---

## 🔍 Why Three Separate Sets?

1. **Training Set:** Model learns from this
   - Large (80%) - lots of examples to learn from
   - Model sees this repeatedly

2. **Validation Set:** Pick best model
   - Medium (10%) - unseen during training
   - Used to stop early or pick best epoch
   - Prevents overfitting

3. **Test Set:** Final unbiased evaluation
   - Small (10%) - completely untouched
   - Only used once at the end
   - Gives honest performance estimate

---

## 💡 Real-World Analogy

**Training:** Student practices with homework problems
- Sees solutions, learns patterns
- Gets feedback, improves

**Validation:** Teacher checks progress on quiz
- New problems (but teacher can adjust)
- Used to see if student is ready

**Testing:** Final exam
- Completely new problems
- Student has no feedback
- True measure of knowledge

---

## 📈 What You'll See in Output

```
Epoch [1/50]
  Train Loss: 0.8234 | Train Acc: 58.23%  ← Training phase
  Val Acc: 59.45%                          ← Validation phase
  ✓ Saved best model                       ← If best validation so far

Epoch [5/50]
  Train Loss: 0.7123 | Train Acc: 65.12%
  Val Acc: 64.89%

...

================================================================================
FINAL EVALUATION                              ← Testing phase (only once!)
================================================================================

Test Accuracy: 67.23%                        ← Final test results
Validation Accuracy: 66.45%
```

---

## 🎯 Bottom Line

- **Training:** Model learns (50 epochs)
- **Validation:** Picks best model (every epoch)
- **Testing:** Final report card (once at end)

The model never sees test data until the very end - this ensures an unbiased evaluation!

