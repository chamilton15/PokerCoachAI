# Data Output Locations & Formats

## 📁 Where Data Gets Saved

All outputs are saved in the **`ml_pipeline/` directory** (same folder as the scripts).

---

## Step 1: Extract Winners

### Output Location:
```
/Users/sethfgn/Desktop/DL_Poker_Project/Github_Repo/PokerCoachAI/ml_pipeline/winners_list.json
```

### Format: JSON

```json
{
  "winners": [
    "player_name_1",
    "player_name_2",
    ...
  ],
  "winner_stats": {
    "player_name_1": {
      "bb_per_hand": 2.45,
      "hands_played": 150,
      "total_winnings": 183.75,
      "big_blind": 0.50
    },
    "player_name_2": {
      "bb_per_hand": 1.82,
      "hands_played": 89,
      "total_winnings": 81.00,
      "big_blind": 0.50
    }
  },
  "total_valid_players": 1250,
  "top_percentile": 20,
  "min_hands": 20
}
```

**Size:** Small (few KB to few MB, depends on number of winners)

---

## Step 2: Extract Training Data

### Output Location:
```
/Users/sethfgn/Desktop/DL_Poker_Project/Github_Repo/PokerCoachAI/ml_pipeline/training_data.json
```

### Format: JSON (Large file!)

```json
{
  "training_data": [
    {
      "hand_id": 123,
      "winner": "player_name",
      "state": {
        "current_state": {
          "hole_cards": ["Ah", "Kh"],
          "hand_score": 8,
          "hand_type": "AKo",
          "position": "BTN",
          "street": "preflop",
          "board_cards": [],
          "pot_size": 2.25,
          "stack_size": 50.0,
          "num_opponents": 2,
          "facing_action": "none"
        },
        "action_history": [
          {
            "player_idx": 0,
            "action_type": "raise",
            "raw_action": "cbr",
            "amount": 1.5,
            "raw_string": "p1 cbr 1.5",
            "street": "preflop",
            "sequence_index": 0,
            "pot_size_before": 0.75,
            "pot_size_after": 2.25
          },
          {
            "player_idx": 1,
            "action_type": "call",
            "raw_action": "cc",
            "amount": 1.5,
            "raw_string": "p2 cc",
            "street": "preflop",
            "sequence_index": 1,
            "pot_size_before": 2.25,
            "pot_size_after": 2.25
          }
        ],
        "history_length": 2,
        "num_raises_in_history": 1,
        "num_calls_in_history": 1,
        "num_folds_in_history": 0,
        "last_action": {
          "player_idx": 1,
          "action_type": "call",
          ...
        }
      },
      "label": "raise",
      "raw_action": "cbr",
      "sequence_position": 2
    },
    ... (thousands more examples)
  ],
  "num_examples": 50000,
  "winners": ["player1", "player2", ...],
  "metadata": {
    "total_winners": 250,
    "handhq_directory": "/Users/.../data/handhq"
  }
}
```

**Size:** Large (10-100+ MB, depends on number of examples)

**Structure:**
- Each item in `training_data` = one decision point
- Contains full state + complete action history
- Label = what winner actually did

---

## Step 3: Prepare Training Dataset

### Output Location:
```
/Users/sethfgn/Desktop/DL_Poker_Project/Github_Repo/PokerCoachAI/ml_pipeline/dataset/
```

### Output Files:

#### 1. `dataset.pkl` (Main dataset file)

**Format:** Python Pickle (binary file)

**Structure:** Dictionary with numpy arrays

```python
{
    'train': {
        'X_current': numpy.ndarray,      # Shape: (N_train, 11)
        'X_history': numpy.ndarray,      # Shape: (N_train, 20, 5)
        'y': numpy.ndarray,              # Shape: (N_train,)
        'y_one_hot': numpy.ndarray       # Shape: (N_train, 3)
    },
    'val': {
        'X_current': numpy.ndarray,      # Shape: (N_val, 11)
        'X_history': numpy.ndarray,      # Shape: (N_val, 20, 5)
        'y': numpy.ndarray,              # Shape: (N_val,)
        'y_one_hot': numpy.ndarray       # Shape: (N_val, 3)
    },
    'test': {
        'X_current': numpy.ndarray,      # Shape: (N_test, 11)
        'X_history': numpy.ndarray,      # Shape: (N_test, 20, 5)
        'y': numpy.ndarray,              # Shape: (N_test,)
        'y_one_hot': numpy.ndarray       # Shape: (N_test, 3)
    },
    'metadata': {
        'num_classes': 3,
        'current_feature_dim': 11,
        'history_max_length': 20,
        'history_feature_dim': 5,
        'label_map': {
            'fold': 0,
            'call': 1,
            'raise': 2
        }
    }
}
```

**Size:** Medium-Large (depends on examples, but compressed by pickle)

**How to Load:**
```python
import pickle

with open('dataset/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

X_train_current = dataset['train']['X_current']
X_train_history = dataset['train']['X_history']
y_train = dataset['train']['y_one_hot']
```

---

#### 2. `dataset_metadata.json` (Human-readable metadata)

**Format:** JSON

```json
{
  "num_train": 40000,
  "num_val": 5000,
  "num_test": 5000,
  "metadata": {
    "num_classes": 3,
    "current_feature_dim": 11,
    "history_max_length": 20,
    "history_feature_dim": 5,
    "label_map": {
      "fold": 0,
      "call": 1,
      "raise": 2
    }
  }
}
```

**Purpose:** Quick reference for dataset info without loading pickle file

---

## 📊 File Sizes (Estimated)

Based on typical dataset sizes:

| File | Estimated Size | Format |
|------|---------------|--------|
| `winners_list.json` | 10 KB - 1 MB | JSON (text) |
| `training_data.json` | 10 MB - 500 MB | JSON (text) |
| `dataset/dataset.pkl` | 5 MB - 200 MB | Pickle (binary) |
| `dataset/dataset_metadata.json` | <1 KB | JSON (text) |

**Note:** Sizes depend heavily on:
- Number of winners found
- Number of hands per winner
- Number of decisions per hand

---

## 📂 Complete Directory Structure After Running

```
ml_pipeline/
├── extract_winners.py
├── extract_training_data.py
├── prepare_training_dataset.py
│
├── winners_list.json                    ← Step 1 output
├── training_data.json                   ← Step 2 output
│
└── dataset/                             ← Step 3 output directory
    ├── dataset.pkl                      ← Main dataset (binary)
    └── dataset_metadata.json            ← Metadata (human-readable)
```

---

## 🔍 Viewing the Data

### View JSON files:
```bash
# View winners
cat ml_pipeline/winners_list.json | python -m json.tool | head -50

# View sample training data
cat ml_pipeline/training_data.json | python -m json.tool | head -100
```

### View Pickle file:
```python
import pickle
import numpy as np

# Load dataset
with open('ml_pipeline/dataset/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Check shapes
print("Train shapes:")
print(f"  X_current: {dataset['train']['X_current'].shape}")
print(f"  X_history: {dataset['train']['X_history'].shape}")
print(f"  y_one_hot: {dataset['train']['y_one_hot'].shape}")

# View metadata
print("\nMetadata:")
print(dataset['metadata'])
```

---

## 💾 Loading Data for Training

### For Neural Network Training:

```python
import pickle
import numpy as np

# Load dataset
with open('ml_pipeline/dataset/dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract training data
X_current_train = data['train']['X_current']  # (N, 11)
X_history_train = data['train']['X_history']  # (N, 20, 5)
y_train = data['train']['y_one_hot']          # (N, 3)

# Extract validation data
X_current_val = data['val']['X_current']
X_history_val = data['val']['X_history']
y_val = data['val']['y_one_hot']

# Extract test data
X_current_test = data['test']['X_current']
X_history_test = data['test']['X_history']
y_test = data['test']['y_one_hot']

# Metadata
num_classes = data['metadata']['num_classes']  # 3
label_map = data['metadata']['label_map']      # {'fold': 0, 'call': 1, 'raise': 2}
```

---

## 📝 Summary

**All files save to:** `ml_pipeline/` directory

**Step 1 →** `winners_list.json` (JSON)
**Step 2 →** `training_data.json` (JSON, large!)
**Step 3 →** `dataset/dataset.pkl` (Pickle, numpy arrays)

**Formats:**
- JSON: Human-readable, easy to inspect
- Pickle: Binary, efficient for numpy arrays, Python-only

**All paths are relative to the script location, so everything stays in `ml_pipeline/` folder!**


