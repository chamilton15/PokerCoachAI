#!/usr/bin/env python3
"""
Step 3: Prepare Training Dataset - Feature Encoding and Label Preparation
Convert extracted training data into neural network-ready format:
- Encode all features numerically
- Create fixed-length action history sequences
- Prepare labels for classification
"""

import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import pickle

# NumPy replacement - using lists instead
# Users can convert to numpy arrays later if needed
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Create a simple numpy-like interface using lists
    class np_array:
        @staticmethod
        def array(data, dtype=None):
            if dtype == 'float32':
                return [[float(x) for x in row] if isinstance(row, list) else float(row) for row in data]
            elif dtype == 'int32':
                return [[int(x) for x in row] if isinstance(row, list) else int(row) for row in data]
            else:
                return data
        @staticmethod
        def zeros(shape, dtype=None):
            if isinstance(shape, int):
                return [0.0] * shape if dtype == 'float32' else [0] * shape
            elif isinstance(shape, tuple):
                if len(shape) == 1:
                    return [0.0] * shape[0] if dtype == 'float32' else [0] * shape[0]
                else:
                    return [[0.0] * shape[1] for _ in range(shape[0])] if dtype == 'float32' else [[0] * shape[1] for _ in range(shape[0])]
    
    class np:
        array = staticmethod(np_array.array)
        zeros = staticmethod(np_array.zeros)
        float32 = 'float32'
        int32 = 'int32'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_coach.hand_strength import HandStrengthEvaluator


class FeatureEncoder:
    """Encode poker game states and action history into numerical features"""
    
    def __init__(self):
        self.evaluator = HandStrengthEvaluator()
        
        # Position encoding
        self.position_map = {
            'UTG': 0, 'UTG+1': 1, 'MP': 2, 'MP+1': 3,
            'HJ': 4, 'CO': 5, 'BTN': 6, 'SB': 7, 'BB': 8
        }
        
        # Action encoding
        self.action_map = {
            'fold': 0,
            'call': 1,
            'raise': 2,
            'showdown': 3
        }
        
        # Reverse map for decoding
        self.action_decode = {v: k for k, v in self.action_map.items()}
    
    def encode_position(self, position: str) -> int:
        """Encode position to integer"""
        return self.position_map.get(position, 4)  # Default to HJ
    
    def encode_action(self, action: str) -> int:
        """Encode action type to integer"""
        return self.action_map.get(action, 1)  # Default to call
    
    def encode_cards(self, cards: List[str] = None) -> Tuple[float, int]:
        """Encode hole cards"""
        if cards is None or len(cards) != 2:
            return 5.0, 0  # Default medium strength, unknown type
        
        hand_score, hand_type = self.evaluator.evaluate_preflop_strength(cards)
        
        # Encode hand type
        hand_type_map = {
            'pair': 1,
            'suited': 2,
            'offsuit': 3,
            'unknown': 0
        }
        type_code = hand_type_map.get(hand_type[:6] if hand_type else 'unknown', 0)
        
        return hand_score, type_code
    
    def encode_action_history(self, history: List[Dict], max_length: int = 20) -> List[List[float]]:
        """
        Encode action history to fixed-length sequence
        
        Args:
            history: List of action dictionaries
            max_length: Maximum sequence length
        
        Returns:
            List of lists (max_length, feature_dim) - compatible with numpy arrays
        """
        # Features per action: [player_idx_norm, action_type, amount_norm, street, pot_norm]
        encoded_history = []
        
        for action in history:
            player_idx = action.get('player_idx', 0)
            action_type = self.encode_action(action.get('action_type', 'call'))
            amount = action.get('amount', 0.0) / 0.50  # Normalize to BB
            street = self.encode_street(action.get('street', 'preflop'))
            pot = action.get('pot_size_after', 1.0) / 0.50  # Normalize to BB
            
            encoded_action = [
                player_idx / 8.0,  # Normalize player index (assuming max 9 players)
                action_type / 3.0,  # Normalize action
                amount / 100.0,     # Normalize bet amount (cap at 100 BB)
                street / 3.0,       # Normalize street (0=preflop, 1=flop, 2=turn, 3=river)
                pot / 200.0         # Normalize pot size (cap at 200 BB)
            ]
            encoded_history.append(encoded_action)
        
        # Pad or truncate to fixed length
        if len(encoded_history) > max_length:
            encoded_history = encoded_history[-max_length:]
        else:
            while len(encoded_history) < max_length:
                encoded_history.insert(0, [0.0, 0.0, 0.0, 0.0, 0.0])  # Padding
        
        # Return as list (can be converted to numpy array later if needed)
        return encoded_history
    
    def encode_street(self, street: str) -> int:
        """Encode street (preflop, flop, turn, river)"""
        street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        return street_map.get(street, 0)
    
    def encode_full_state(self, training_example: Dict) -> Tuple[List[float], List[List[float]]]:
        """
        Encode full state with current features and action history
        
        Returns:
            (current_features, action_history_sequence)
        """
        state = training_example['state']
        current = state['current_state']
        history = state['action_history']
        
        # Encode current state features
        hand_score, hand_type_code = self.encode_cards(current.get('hole_cards'))
        position = self.encode_position(current.get('position', 'MP'))
        
        current_features = [
            hand_score / 10.0,                              # Hand strength (0-1)
            hand_type_code / 3.0,                           # Hand type (normalized)
            position / 8.0,                                 # Position (normalized)
            self.encode_street(current.get('street', 'preflop')) / 3.0,  # Street
            current.get('pot_size', 1.0) / 50.0,          # Pot size (normalized)
            current.get('stack_size', 50.0) / 200.0,       # Stack size (normalized)
            current.get('num_opponents', 1) / 8.0,         # Number of opponents (normalized)
            self.encode_action(current.get('facing_action', 'none')) / 3.0,  # Facing action
            len(history) / 30.0,                           # History length (normalized)
            state.get('num_raises_in_history', 0) / 10.0,  # Raises in history
            state.get('num_calls_in_history', 0) / 10.0,  # Calls in history
        ]
        
        # Encode action history sequence
        history_sequence = self.encode_action_history(history, max_length=20)
        
        return current_features, history_sequence


class LabelEncoder:
    """Encode labels for neural network"""
    
    def __init__(self):
        self.label_map = {
            'fold': 0,
            'call': 1,
            'raise': 2
        }
        self.reverse_map = {v: k for k, v in self.label_map.items()}
    
    def encode_label(self, label: str) -> int:
        """Encode action label to integer"""
        return self.label_map.get(label, 1)  # Default to call
    
    def decode_label(self, encoded: int) -> str:
        """Decode integer back to action"""
        return self.reverse_map.get(encoded, 'call')
    
    def to_one_hot(self, label: int, num_classes: int = 3) -> List[float]:
        """Convert label to one-hot encoding"""
        one_hot = [0.0] * num_classes
        if 0 <= label < num_classes:
            one_hot[label] = 1.0
        return one_hot


def prepare_dataset(training_data_file: str = "training_data.json",
                   output_dir: str = "dataset") -> Dict[str, Any]:
    """
    Prepare dataset from extracted training data
    
    Returns:
        Dictionary with encoded features and labels
    """
    
    print("="*70)
    print("STEP 3: PREPARING TRAINING DATASET")
    print("="*70)
    
    # Load training data
    data_path = Path(__file__).parent / training_data_file
    if not data_path.exists():
        print(f"ERROR: {training_data_file} not found. Run extract_training_data.py first!")
        return None
    
    print(f"\nLoading training data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    training_examples = data['training_data']
    print(f"Loaded {len(training_examples)} training examples")
    
    # Initialize encoders
    feature_encoder = FeatureEncoder()
    label_encoder = LabelEncoder()
    
    # Process examples
    current_features_list = []
    history_sequences_list = []
    labels_list = []
    one_hot_labels_list = []
    
    print("\nEncoding features and labels...")
    skipped = 0
    
    for i, example in enumerate(training_examples):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(training_examples)} examples...")
        
        try:
            # Encode features
            current_features, history_sequence = feature_encoder.encode_full_state(example)
            
            # Encode label
            label_str = example['label']
            label_int = label_encoder.encode_label(label_str)
            label_one_hot = label_encoder.to_one_hot(label_int)
            
            current_features_list.append(current_features)
            history_sequences_list.append(history_sequence)
            labels_list.append(label_int)
            one_hot_labels_list.append(label_one_hot)
        
        except Exception as e:
            skipped += 1
            continue
    
    if skipped > 0:
        print(f"  Skipped {skipped} examples due to errors")
    
    # Convert to arrays (lists, compatible with numpy arrays)
    print("\nPreparing data arrays...")
    X_current = current_features_list
    X_history = history_sequences_list
    y_labels = labels_list
    y_one_hot = one_hot_labels_list
    
    # Convert to numpy arrays if available, otherwise keep as lists
    if HAS_NUMPY:
        X_current = np.array(X_current, dtype=np.float32)
        X_history = np.array(X_history, dtype=np.float32)
        y_labels = np.array(y_labels, dtype=np.int32)
        y_one_hot = np.array(y_one_hot, dtype=np.float32)
    
    print(f"\nDataset shape:")
    if HAS_NUMPY:
        print(f"  Current features: {X_current.shape}")
        print(f"  History sequences: {X_history.shape}")
        print(f"  Labels: {y_labels.shape}")
        print(f"  One-hot labels: {y_one_hot.shape}")
    else:
        print(f"  Current features: {len(X_current)} examples, {len(X_current[0]) if X_current else 0} features")
        print(f"  History sequences: {len(X_history)} examples, {len(X_history[0]) if X_history else 0} actions, {len(X_history[0][0]) if X_history and X_history[0] else 0} features per action")
        print(f"  Labels: {len(y_labels)} examples")
        print(f"  One-hot labels: {len(y_one_hot)} examples, {len(y_one_hot[0]) if y_one_hot else 0} classes")
    
    # Label distribution
    from collections import Counter
    label_counts = Counter(labels_list)
    print(f"\nLabel distribution:")
    for label_int, count in sorted(label_counts.items()):
        label_str = label_encoder.decode_label(label_int)
        print(f"  {label_str:10s}: {count:6d} ({count/len(labels_list)*100:5.1f}%)")
    
    # Split dataset
    print("\nSplitting dataset (80/10/10)...")
    n_total = len(X_current)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    # Shuffle indices
    indices = list(range(n_total))
    random.shuffle(indices)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Split data (works for both lists and numpy arrays)
    if HAS_NUMPY:
        dataset = {
            'train': {
                'X_current': X_current[train_idx],
                'X_history': X_history[train_idx],
                'y': y_labels[train_idx],
                'y_one_hot': y_one_hot[train_idx]
            },
            'val': {
                'X_current': X_current[val_idx],
                'X_history': X_history[val_idx],
                'y': y_labels[val_idx],
                'y_one_hot': y_one_hot[val_idx]
            },
            'test': {
                'X_current': X_current[test_idx],
                'X_history': X_history[test_idx],
                'y': y_labels[test_idx],
                'y_one_hot': y_one_hot[test_idx]
            },
            'metadata': {
                'num_classes': 3,
                'current_feature_dim': int(X_current.shape[1]),
                'history_max_length': int(X_history.shape[1]),
                'history_feature_dim': int(X_history.shape[2]),
                'label_map': label_encoder.label_map
            }
        }
    else:
        # For lists, use list comprehensions
        dataset = {
            'train': {
                'X_current': [X_current[i] for i in train_idx],
                'X_history': [X_history[i] for i in train_idx],
                'y': [y_labels[i] for i in train_idx],
                'y_one_hot': [y_one_hot[i] for i in train_idx]
            },
            'val': {
                'X_current': [X_current[i] for i in val_idx],
                'X_history': [X_history[i] for i in val_idx],
                'y': [y_labels[i] for i in val_idx],
                'y_one_hot': [y_one_hot[i] for i in val_idx]
            },
            'test': {
                'X_current': [X_current[i] for i in test_idx],
                'X_history': [X_history[i] for i in test_idx],
                'y': [y_labels[i] for i in test_idx],
                'y_one_hot': [y_one_hot[i] for i in test_idx]
            },
            'metadata': {
                'num_classes': 3,
                'current_feature_dim': len(X_current[0]) if X_current else 0,
                'history_max_length': len(X_history[0]) if X_history else 0,
                'history_feature_dim': len(X_history[0][0]) if X_history and X_history[0] else 0,
                'label_map': label_encoder.label_map
            }
        }
    
    # Save dataset
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(exist_ok=True)
    
    dataset_file = output_path / "dataset.pkl"
    print(f"\nSaving dataset to {dataset_file}...")
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset, f)
    
    # Also save metadata as JSON for easy inspection
    metadata_file = output_path / "dataset_metadata.json"
    
    # Get dimensions (works for both numpy arrays and lists)
    if HAS_NUMPY:
        current_dim = int(X_current.shape[1])
        history_length = int(X_history.shape[1])
        history_dim = int(X_history.shape[2])
    else:
        current_dim = len(X_current[0]) if X_current else 0
        history_length = len(X_history[0]) if X_history else 0
        history_dim = len(X_history[0][0]) if X_history and X_history[0] else 0
    
    with open(metadata_file, 'w') as f:
        json.dump({
            'num_train': len(train_idx),
            'num_val': len(val_idx),
            'num_test': len(test_idx),
            'metadata': {
                'num_classes': 3,
                'current_feature_dim': current_dim,
                'history_max_length': history_length,
                'history_feature_dim': history_dim,
                'label_map': label_encoder.label_map
            }
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Dataset prepared and saved!")
    print(f"  Train: {len(train_idx)} examples")
    print(f"  Val: {len(val_idx)} examples")
    print(f"  Test: {len(test_idx)} examples")
    print(f"  Saved to: {dataset_file}")
    print(f"{'='*70}")
    
    return dataset


def main():
    """Main execution"""
    random.seed(42)  # For reproducible splits
    dataset = prepare_dataset()
    return dataset


if __name__ == "__main__":
    dataset = main()


