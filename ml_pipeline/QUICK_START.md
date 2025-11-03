# ML Pipeline - Quick Start Guide

## ✅ What's Been Built

Complete data extraction and labeling pipeline for neural network training:

### Step 1: `extract_winners.py`
- Parses all handhq `.phhs` files
- Calculates BB/hand for every player
- Identifies top 20% winners
- **Output:** `winners_list.json`

### Step 2: `extract_training_data.py`
- Extracts all decision points from winners' hands
- **Includes full action history** up to each decision
- Records what winner actually did (label)
- **Output:** `training_data.json`

### Step 3: `prepare_training_dataset.py`
- Encodes all features numerically
- Creates fixed-length action history sequences
- Prepares one-hot labels
- Splits into train/val/test (80/10/10)
- **Output:** `dataset/dataset.pkl`

---

## 🚀 How to Run

```bash
cd ml_pipeline

# Step 1: Find winners (this will take a while - processes all handhq files)
python extract_winners.py

# Step 2: Extract training data (also takes time)
python extract_training_data.py

# Step 3: Prepare dataset (fast - just encoding)
python prepare_training_dataset.py
```

---

## 📊 Expected Outputs

### After Step 1:
```
winners_list.json
{
  "winners": ["player1", "player2", ...],
  "winner_stats": {
    "player1": {"bb_per_hand": 2.5, "hands_played": 150, ...}
  }
}
```

### After Step 2:
```
training_data.json
{
  "training_data": [
    {
      "hand_id": 123,
      "state": {
        "current_state": {...},
        "action_history": [...]  // Full history!
      },
      "label": "raise"
    },
    ...
  ]
}
```

### After Step 3:
```
dataset/
├── dataset.pkl           # Numpy arrays ready for training
└── dataset_metadata.json  # Dataset info
```

---

## 🎯 Key Features

### Action History Included:
Each training example includes **complete context**:
```python
state = {
    'current_state': {...},
    'action_history': [
        {'player': 0, 'action': 'raise', 'amount': 1.5},
        {'player': 1, 'action': 'call', 'amount': 1.5},
        {'player': 2, 'action': 'fold'},
        # ... all previous actions in hand
    ]
}
```

**This gives the neural network full context for each decision!**

---

## ⏱️ Runtime Estimates

**Step 1 (extract_winners):**
- ~10-30 minutes (depends on dataset size)
- Processes all `.phhs` files
- Single pass through data

**Step 2 (extract_training_data):**
- ~20-60 minutes (depends on number of winners)
- Processes only winners' hands
- Multiple passes (one per winner)

**Step 3 (prepare_training_dataset):**
- ~1-5 minutes
- Fast encoding step
- Just processes JSON file

---

## 🐛 Troubleshooting

### "No .phhs files found"
- Check path: `/Users/sethfgn/Desktop/DL_Poker_Project/Poker_Data_Set/data/handhq`
- Ensure directory structure is correct

### "No valid players found"
- Minimum hands requirement (20) might be too high
- Try reducing `min_hands` parameter in `extract_winners.py`

### "No training data extracted"
- Check that winners were found in Step 1
- Verify winners actually appear in handhq files
- Some winners might only be in specific files

### Memory errors
- Process files in batches
- Increase system memory
- Filter to smaller subset for testing

---

## 📈 Next Steps

After running all 3 steps:
1. ✅ Dataset is ready for neural network
2. ⏳ Build network architecture (LSTM/Transformer)
3. ⏳ Train model on dataset
4. ⏳ Evaluate performance
5. ⏳ Use to analyze new players

---

**Ready to extract data! Start with Step 1.**


