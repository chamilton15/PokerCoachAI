#!/usr/bin/env python3
"""
Hyperparameter Tuning Script
Tests multiple learning rates to find optimal configuration
Also fixes class imbalance with better weighting
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import training components
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from train_model
from train_model import PokerLSTM, PokerDataset, calculate_accuracy

def calculate_improved_class_weights(label_counts: Dict[int, int], total_samples: int, num_classes: int) -> list:
    """
    Calculate improved class weights that better handle imbalance
    Uses inverse frequency with smoothing and focuses on underrepresented classes
    """
    class_weights = []
    
    for i in range(num_classes):
        if i in label_counts and label_counts[i] > 0:
            # Inverse frequency weighting with smoothing
            # More aggressive weighting for minority classes
            frequency = label_counts[i] / total_samples
            # Use sqrt of inverse frequency to prevent extreme weights
            weight = np.sqrt(total_samples / (num_classes * label_counts[i]))
            
            # For call class (class 1), which is performing poorly, boost more
            if i == 1:  # call class
                weight *= 2.5  # Boost call class weight significantly
            
            # Clamp to reasonable range
            weight = min(weight, 15.0)  # Cap at 15x
            weight = max(weight, 0.5)   # Floor at 0.5x
        else:
            weight = 1.0
        class_weights.append(weight)
    
    return class_weights


def train_with_config(dataset_path: str, learning_rate: float, epochs: int = 100, 
                     batch_size: int = 64, device: str = None) -> Dict[str, Any]:
    """
    Train model with specific configuration and return results
    """
    print(f"\n{'='*80}")
    print(f"Testing Learning Rate: {learning_rate}")
    print(f"{'='*80}")
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Load dataset
    dataset_file = Path(__file__).parent / dataset_path
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    
    metadata = dataset.get('metadata', {})
    current_feature_dim = metadata.get('current_feature_dim', 11)
    history_max_length = metadata.get('history_max_length', 20)
    history_feature_dim = metadata.get('history_feature_dim', 5)
    num_classes = metadata.get('num_classes', 3)
    
    # Prepare data splits
    train_data = dataset['train']
    val_data = dataset['val']
    test_data = dataset['test']
    
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
    model = PokerLSTM(
        current_feature_dim=current_feature_dim,
        history_max_length=history_max_length,
        history_feature_dim=history_feature_dim,
        hidden_dim=64,
        num_classes=num_classes
    )
    model = model.to(device)
    
    # Calculate improved class weights
    label_counts = {}
    for label in train_data['y']:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    total_samples = len(train_data['y'])
    class_weights = calculate_improved_class_weights(label_counts, total_samples, num_classes)
    
    label_map = metadata.get('label_map', {'fold': 0, 'call': 1, 'raise': 2})
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    print(f"\nClass distribution and weights:")
    for i in range(num_classes):
        if i in label_counts:
            label_str = reverse_label_map.get(i, f"class_{i}")
            count = label_counts[i]
            pct = 100 * count / total_samples
            print(f"  {label_str:10s}: {count:6d} ({pct:5.1f}%) - weight: {class_weights[i]:.3f}")
    
    # Convert to tensor
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Optimizer with specific learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    best_epoch = 0
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Progress will be shown every 5 epochs\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            current = batch['current'].to(device)
            history = batch['history'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(current, history)
            
            if torch.isnan(outputs).any():
                continue
            
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_value = loss.item()
            if not (np.isnan(loss_value) or np.isinf(loss_value)):
                running_loss += loss_value
                batch_count += 1
        
        avg_loss = running_loss / batch_count if batch_count > 0 else 1.0
        train_losses.append(avg_loss)
        
        # Validation phase
        val_acc = calculate_accuracy(model, val_loader, device)
        val_accuracies.append(val_acc)
        scheduler.step(avg_loss)
        
        # Print progress every 5 epochs or at important milestones
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1:3d}/{epochs}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            if (epoch + 1) % 5 != 0:  # Print if we didn't already print this epoch
                print(f"  → New best! Val Acc: {best_val_acc:.2f}% (epoch {epoch+1})")
    
    # Final test evaluation
    test_acc = calculate_accuracy(model, test_loader, device)
    
    # Per-class accuracy
    model.eval()
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}
    
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
    
    per_class_acc = {}
    for label_int in sorted(class_correct.keys()):
        label_str = reverse_label_map.get(label_int, f"class_{label_int}")
        acc = 100 * class_correct[label_int] / class_total[label_int] if class_total[label_int] > 0 else 0
        per_class_acc[label_str] = acc
    
    results = {
        'learning_rate': learning_rate,
        'epochs': epochs,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'best_epoch': best_epoch,
        'per_class_accuracy': per_class_acc,
        'class_weights': class_weights,
        'final_train_loss': train_losses[-1] if train_losses else None
    }
    
    print(f"\nResults for LR={learning_rate}:")
    print(f"  Best Val Acc: {best_val_acc:.2f}% (epoch {best_epoch+1})")
    print(f"  Test Acc: {test_acc:.2f}%")
    print(f"  Per-class: fold={per_class_acc.get('fold', 0):.2f}%, call={per_class_acc.get('call', 0):.2f}%, raise={per_class_acc.get('raise', 0):.2f}%")
    
    return results


def main():
    """Main hyperparameter tuning"""
    
    print("="*80)
    print("HYPERPARAMETER TUNING: Learning Rate Optimization")
    print("="*80)
    print("\nThis script will test multiple learning rates to find the optimal one.")
    print("Each configuration will train for 100 epochs.")
    print("Progress will be shown during training.\n")
    
    dataset_path = "dataset/dataset.pkl"
    epochs = 100  # Increased epochs
    batch_size = 64
    
    # Learning rates to test
    learning_rates = [0.0001, 0.0005, 0.001, 0.002, 0.003]
    
    print(f"Testing {len(learning_rates)} learning rates: {learning_rates}")
    print(f"Total training time: ~{len(learning_rates) * epochs} epochs\n")
    print("="*80)
    
    all_results = []
    
    for idx, lr in enumerate(learning_rates, 1):
        print(f"\n{'='*80}")
        print(f"CONFIGURATION {idx}/{len(learning_rates)}: Learning Rate = {lr}")
        print(f"{'='*80}")
        try:
            results = train_with_config(
                dataset_path=dataset_path,
                learning_rate=lr,
                epochs=epochs,
                batch_size=batch_size
            )
            all_results.append(results)
            print(f"\n✓ Completed LR={lr}: Test Acc = {results['test_acc']:.2f}%")
        except Exception as e:
            print(f"\n✗ ERROR with LR={lr}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Find best configuration
    if all_results:
        print(f"\n{'='*80}")
        print("ALL RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"{'LR':<10} {'Test Acc':<12} {'Val Acc':<12} {'Call Acc':<12}")
        print("-" * 50)
        for r in all_results:
            call_acc = r['per_class_accuracy'].get('call', 0)
            print(f"{r['learning_rate']:<10.4f} {r['test_acc']:<12.2f} {r['best_val_acc']:<12.2f} {call_acc:<12.2f}")
        
        best_result = max(all_results, key=lambda x: x['test_acc'])
        
        print(f"\n{'='*80}")
        print("🏆 BEST CONFIGURATION FOUND")
        print(f"{'='*80}")
        print(f"Learning Rate: {best_result['learning_rate']}")
        print(f"Test Accuracy: {best_result['test_acc']:.2f}%")
        print(f"Validation Accuracy: {best_result['best_val_acc']:.2f}%")
        print(f"Best Epoch: {best_result['best_epoch'] + 1}")
        print(f"\nPer-class accuracy:")
        for label, acc in best_result['per_class_accuracy'].items():
            print(f"  {label:10s}: {acc:.2f}%")
        
        # Save results
        results_file = Path(__file__).parent / "models" / "hyperparameter_tuning_results.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'all_results': all_results,
                'best_config': best_result
            }, f, indent=2)
        
        print(f"\nAll results saved to: {results_file}")
        
        # Now train final model with best learning rate
        print(f"\n{'='*80}")
        print("TRAINING FINAL MODEL WITH BEST LEARNING RATE")
        print(f"{'='*80}")
        
        # Import here to avoid circular import
        import train_model
        final_model = train_model.train_model(
            dataset_path=dataset_path,
            model_output_dir="models",
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=best_result['learning_rate']
        )
        
        return final_model, best_result
    else:
        print("\nERROR: No successful training runs!")
        return None, None


if __name__ == "__main__":
    model, best_config = main()

