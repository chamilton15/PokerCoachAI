#!/usr/bin/env python3
"""
Train LSTM Neural Network for Poker Decision Prediction
Uses trained model to predict optimal actions (fold/call/raise)
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List
import random

# Try to import ML libraries
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError as e:
    HAS_NUMPY = False
    print("="*80)
    print("ERROR: NumPy not available!")
    print("="*80)
    print("\nSolution:")
    print("1. Make sure you're using conda Python (not system Python 3.9)")
    print("2. Install PyTorch: pip install torch")
    print("3. Run with: /Users/sethfgn/miniforge3/bin/python train_model.py")
    print("\nOr activate conda environment:")
    print("  conda activate base")
    print("  python train_model.py")
    print(f"\nError details: {e}")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch not installed. Please install: pip install torch")
    print("Alternatively, we can create a TensorFlow/Keras version")
    sys.exit(1)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_coach.hand_strength import HandStrengthEvaluator


class PokerDataset(Dataset):
    """PyTorch Dataset for poker training data"""
    
    def __init__(self, X_current, X_history, y_labels):
        # Convert to numpy arrays if they're lists
        if isinstance(X_current, list):
            X_current = np.array(X_current, dtype=np.float32)
        if isinstance(X_history, list):
            X_history = np.array(X_history, dtype=np.float32)
        if isinstance(y_labels, list):
            y_labels = np.array(y_labels, dtype=np.int64)
        
        # Check for NaN/Inf and replace
        if np.isnan(X_current).any() or np.isinf(X_current).any():
            print("WARNING: NaN/Inf found in X_current, replacing with 0")
            X_current = np.nan_to_num(X_current, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if np.isnan(X_history).any() or np.isinf(X_history).any():
            print("WARNING: NaN/Inf found in X_history, replacing with 0")
            X_history = np.nan_to_num(X_history, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize to prevent extreme values
        X_current = np.clip(X_current, -10.0, 10.0)
        X_history = np.clip(X_history, -10.0, 10.0)
        
        self.X_current = torch.FloatTensor(X_current)
        self.X_history = torch.FloatTensor(X_history)
        self.y = torch.LongTensor(y_labels)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {
            'current': self.X_current[idx],
            'history': self.X_history[idx],
            'label': self.y[idx]
        }


class PokerLSTM(nn.Module):
    """LSTM-based neural network for poker action prediction"""
    
    def __init__(self, current_feature_dim=11, history_max_length=20, 
                 history_feature_dim=5, hidden_dim=64, num_classes=3):
        super(PokerLSTM, self).__init__()
        
        self.current_feature_dim = current_feature_dim
        self.history_max_length = history_max_length
        self.history_feature_dim = history_feature_dim
        
        # Process current state features
        self.current_encoder = nn.Sequential(
            nn.Linear(current_feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Process action history sequence with LSTM
        self.lstm = nn.LSTM(
            input_size=history_feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=False
        )
        
        # Combine current state and history
        self.combiner = nn.Sequential(
            nn.Linear(32 + hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduced dropout
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),  # Reduced dropout
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use smaller initialization to prevent exploding gradients
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        # Input-to-hidden weights
                        nn.init.xavier_uniform_(param.data, gain=0.5)
                    elif 'weight_hh' in name:
                        # Hidden-to-hidden weights
                        nn.init.orthogonal_(param.data, gain=0.5)
                    elif 'bias' in name:
                        # Initialize bias to small values
                        nn.init.constant_(param.data, 0.0)
                        # Set forget gate bias to 1 (helps with gradient flow)
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1.0)
        
    def forward(self, current_state, history_sequence):
        """
        Forward pass
        
        Args:
            current_state: (batch_size, current_feature_dim)
            history_sequence: (batch_size, history_max_length, history_feature_dim)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # Encode current state
        current_encoded = self.current_encoder(current_state)  # (batch, 32)
        
        # Process history sequence with LSTM
        lstm_out, (hidden, cell) = self.lstm(history_sequence)  # lstm_out: (batch, seq_len, hidden_dim)
        history_encoded = lstm_out[:, -1, :]  # Take last output (batch, hidden_dim)
        
        # Concatenate current and history features
        combined = torch.cat([current_encoded, history_encoded], dim=1)  # (batch, 32 + hidden_dim)
        
        # Final prediction
        logits = self.combiner(combined)  # (batch, num_classes)
        
        return logits


def calculate_accuracy(model, dataloader, device):
    """Calculate accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            current = batch['current'].to(device)
            history = batch['history'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(current, history)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total if total > 0 else 0.0


def train_model(dataset_path: str = "dataset/dataset.pkl",
                model_output_dir: str = "models",
                epochs: int = 50,
                batch_size: int = 64,
                learning_rate: float = 0.001,
                device: str = None):
    """
    Train the LSTM model
    
    Args:
        dataset_path: Path to prepared dataset
        model_output_dir: Directory to save model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: 'cuda', 'cpu', or None (auto-detect)
    """
    
    print("="*80)
    print("TRAINING LSTM NEURAL NETWORK FOR POKER DECISION PREDICTION")
    print("="*80)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"\nUsing device: {device}")
    
    # Load dataset
    dataset_file = Path(__file__).parent / dataset_path
    if not dataset_file.exists():
        print(f"ERROR: Dataset file not found: {dataset_file}")
        print("Please run prepare_training_dataset.py first!")
        return None
    
    print(f"\nLoading dataset from {dataset_file}...")
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    
    # Get metadata
    metadata = dataset.get('metadata', {})
    current_feature_dim = metadata.get('current_feature_dim', 11)
    history_max_length = metadata.get('history_max_length', 20)
    history_feature_dim = metadata.get('history_feature_dim', 5)
    num_classes = metadata.get('num_classes', 3)
    
    print(f"\nDataset dimensions:")
    print(f"  Current features: {current_feature_dim}")
    print(f"  History sequence: {history_max_length} actions × {history_feature_dim} features")
    print(f"  Number of classes: {num_classes}")
    
    # Prepare data splits
    train_data = dataset['train']
    val_data = dataset['val']
    test_data = dataset['test']
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_data['X_current'])} examples")
    print(f"  Validation: {len(val_data['X_current'])} examples")
    print(f"  Test: {len(test_data['X_current'])} examples")
    
    # Create datasets
    train_dataset = PokerDataset(
        train_data['X_current'],
        train_data['X_history'],
        train_data['y']
    )
    val_dataset = PokerDataset(
        val_data['X_current'],
        val_data['X_history'],
        val_data['y']
    )
    test_dataset = PokerDataset(
        test_data['X_current'],
        test_data['X_history'],
        test_data['y']
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    print(f"\nInitializing LSTM model...")
    model = PokerLSTM(
        current_feature_dim=current_feature_dim,
        history_max_length=history_max_length,
        history_feature_dim=history_feature_dim,
        hidden_dim=64,
        num_classes=num_classes
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Calculate class weights to handle imbalance
    label_counts = {}
    for label in train_data['y']:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    total_samples = len(train_data['y'])
    num_classes = len(label_counts)
    
    print(f"\nClass distribution in training set:")
    label_map = metadata.get('label_map', {'fold': 0, 'call': 1, 'raise': 2})
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    # Calculate inverse frequency weights (handle class imbalance)
    # Use balanced approach - don't let weights get too extreme
    class_weights = []
    for i in range(num_classes):
        if i in label_counts and label_counts[i] > 0:
            # Balanced weight: total / (num_classes * count)
            # Clamp to prevent extreme values
            weight = total_samples / (num_classes * label_counts[i])
            weight = min(weight, 10.0)  # Cap at 10x to prevent extreme weights
            weight = max(weight, 0.1)   # Floor at 0.1x
        else:
            weight = 1.0
        class_weights.append(weight)
        if i in label_counts:
            label_str = reverse_label_map.get(i, f"class_{i}")
            count = label_counts[i]
            pct = 100 * count / total_samples
            print(f"  {label_str:10s}: {count:6d} ({pct:5.1f}%) - weight: {class_weights[i]:.3f}")
    
    # Check for NaN/Inf in weights
    if any(np.isnan(w) or np.isinf(w) for w in class_weights):
        print("WARNING: Invalid class weights detected! Using uniform weights instead.")
        class_weights = [1.0] * num_classes
    
    # Convert to tensor for loss function
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"\nUsing class weights: {class_weights}")
    
    # Try balanced loss first, fallback to regular if needed
    try:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    except Exception as e:
        print(f"WARNING: Could not use weighted loss: {e}")
        print("Falling back to unweighted loss")
        criterion = nn.CrossEntropyLoss()
    
    # Lower learning rate for more stable training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}")
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            current = batch['current'].to(device)
            history = batch['history'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(current, history)
            
            # Check for NaN in outputs before loss calculation
            if torch.isnan(outputs).any():
                print(f"WARNING: NaN in model outputs at batch {batch_idx}, skipping")
                continue
            
            loss = criterion(outputs, labels)
            
            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss at batch {batch_idx}, skipping")
                continue
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding gradients
            optimizer.step()
            
            # Statistics
            loss_value = loss.item()
            if not (np.isnan(loss_value) or np.isinf(loss_value)):
                running_loss += loss_value
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Only calculate average if we have valid losses
        valid_batches = max(1, len(train_loader))  # Avoid division by zero
        if running_loss > 0 and not np.isnan(running_loss) and not np.isinf(running_loss):
            avg_loss = running_loss / valid_batches
        else:
            avg_loss = 1.0  # Use 1.0 instead of nan/inf for stability
            if epoch == 0:
                print("WARNING: Loss calculation issues detected, using fallback value")
        
        train_acc = 100 * train_correct / train_total
        train_losses.append(avg_loss if not np.isnan(avg_loss) else float('inf'))
        
        # Validation phase
        val_acc = calculate_accuracy(model, val_loader, device)
        val_accuracies.append(val_acc)
        scheduler.step(avg_loss)  # Reduce LR on plateau
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            loss_str = f"{avg_loss:.4f}" if not (np.isnan(avg_loss) or np.isinf(avg_loss)) else "nan"
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {loss_str} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Acc: {val_acc:.2f}%")
            
            # Debug: Check prediction distribution
            if epoch == 0 or (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    sample_batch = next(iter(train_loader))
                    sample_outputs = model(sample_batch['current'].to(device), 
                                          sample_batch['history'].to(device))
                    sample_probs = torch.softmax(sample_outputs, dim=1)
                    avg_probs = sample_probs.mean(dim=0).cpu().numpy()
                    print(f"  Sample predictions: fold={avg_probs[0]:.3f}, call={avg_probs[1]:.3f}, raise={avg_probs[2]:.3f}")
                model.train()
            print()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_output_path = Path(__file__).parent / model_output_dir
            model_output_path.mkdir(exist_ok=True)
            
            model_file = model_output_path / "poker_coach_model.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_architecture': {
                    'current_feature_dim': current_feature_dim,
                    'history_max_length': history_max_length,
                    'history_feature_dim': history_feature_dim,
                    'hidden_dim': 64,
                    'num_classes': num_classes
                },
                'metadata': metadata,
                'epoch': epoch,
                'val_acc': val_acc,
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_file)
            print(f"  ✓ Saved best model (val_acc: {best_val_acc:.2f}%)")
    
    # Final evaluation on test set
    print(f"\n{'='*80}")
    print("FINAL EVALUATION")
    print(f"{'='*80}")
    
    test_acc = calculate_accuracy(model, test_loader, device)
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Validation Accuracy: {best_val_acc:.2f}%")
    
    # Baseline comparison (random and majority class)
    # Reuse label_counts from earlier if available, otherwise recalculate
    if 'label_counts' not in locals():
        label_counts = {}
        for label in train_data['y']:
            label_counts[label] = label_counts.get(label, 0) + 1
    
    majority_class = max(label_counts, key=label_counts.get)
    majority_count = label_counts[majority_class]
    majority_baseline = 100 * majority_count / len(train_data['y'])
    random_baseline = 100 / num_classes
    
    print(f"\nBaseline Comparisons:")
    print(f"  Random baseline (1/{num_classes}): {random_baseline:.2f}%")
    print(f"  Majority class baseline: {majority_baseline:.2f}% (class {majority_class})")
    print(f"  Model improvement over random: +{test_acc - random_baseline:.2f}%")
    print(f"  Model improvement over majority: +{test_acc - majority_baseline:.2f}%")
    
    # Per-class accuracy
    print(f"\nPer-class accuracy on test set:")
    model.eval()
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}
    
    label_map = metadata.get('label_map', {'fold': 0, 'call': 1, 'raise': 2})
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    with torch.no_grad():
        for batch in test_loader:
            current = batch['current'].to(device)
            history = batch['history'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(current, history)
            _, predicted = torch.max(outputs.data, 1)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    for label_int in sorted(class_correct.keys()):
        label_str = reverse_label_map.get(label_int, f"class_{label_int}")
        acc = 100 * class_correct[label_int] / class_total[label_int] if class_total[label_int] > 0 else 0
        print(f"  {label_str:10s}: {acc:6.2f}% ({class_correct[label_int]}/{class_total[label_int]})")
    
    # Save training history
    history_file = model_output_path / "training_history.json"
    with open(history_file, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'final_test_acc': test_acc,
            'best_val_acc': best_val_acc,
            'baselines': {
                'random': random_baseline,
                'majority_class': majority_baseline
            },
            'per_class_accuracy': {
                reverse_label_map.get(k, f"class_{k}"): 100 * class_correct[k] / class_total[k] 
                if class_total[k] > 0 else 0
                for k in sorted(class_correct.keys())
            }
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Model saved to: {model_file}")
    print(f"Training history saved to: {history_file}")
    print(f"{'='*80}")
    
    return model


def main():
    """Main execution"""
    
    # Configuration
    dataset_path = "dataset/dataset.pkl"
    model_output_dir = "models"
    epochs = 50
    batch_size = 32  # Smaller batch size for more stable gradients
    learning_rate = 0.0001  # Even lower learning rate for stability
    
    # Train model
    model = train_model(
        dataset_path=dataset_path,
        model_output_dir=model_output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    return model


if __name__ == "__main__":
    model = main()

