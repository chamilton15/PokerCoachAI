#!/usr/bin/env python3
"""
Model Inference for Poker Coach AI
Analyzes player sessions using the trained model
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_coach.parser import parse_hand_history, Hand
from ml_pipeline.train_model import PokerLSTM, PokerDataset
from ml_pipeline.prepare_training_dataset import FeatureEncoder
from ml_pipeline.extract_training_data import parse_action_string, extract_current_state, extract_action_history

def load_trained_model(model_path: str = "models/poker_coach_model.pt"):
    """Load the trained model"""
    if Path(model_path).is_absolute():
        model_file = Path(model_path)
    else:
        # If relative, assume it's relative to ml_pipeline directory
        model_file = Path(__file__).parent / model_path
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    checkpoint = torch.load(model_file, map_location='cpu')
    arch = checkpoint['model_architecture']
    
    model = PokerLSTM(
        current_feature_dim=arch['current_feature_dim'],
        history_max_length=arch['history_max_length'],
        history_feature_dim=arch['history_feature_dim'],
        hidden_dim=arch['hidden_dim'],
        num_classes=arch['num_classes']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['metadata']

def extract_player_decisions(hands: List[Hand], player_name: str) -> List[Dict[str, Any]]:
    """Extract decision points for a specific player"""
    decisions = []
    
    for hand in hands:
        # Find player index - try exact match first, then partial
        player_idx = None
        for i, player in enumerate(hand.players):
            if player == player_name:
                player_idx = i
                break
        
        # If not found, try case-insensitive partial match
        if player_idx is None:
            for i, player in enumerate(hand.players):
                if player_name.lower() in player.lower() or player.lower() in player_name.lower():
                    player_idx = i
                    break
        
        if player_idx is None:
            continue
        
        # Extract action history
        action_history = []
        player_marker = f'p{player_idx + 1}'
        street = 'preflop'
        
        for i, action_str in enumerate(hand.actions):
            # Check for street changes
            if action_str.startswith('d db'):
                if street == 'preflop':
                    street = 'flop'
                elif street == 'flop':
                    street = 'turn'
                elif street == 'turn':
                    street = 'river'
                continue  # Skip dealing actions
            
            # Parse action - only process player actions (start with 'p' followed by number)
            if not action_str.startswith('p') or len(action_str) < 2 or not action_str[1].isdigit():
                continue
            
            try:
                parsed = parse_action_string(action_str)
                if parsed:
                    parsed['street'] = street
                    action_history.append(parsed)
                    
                    # Check if this is player's decision point
                    if action_str.startswith(player_marker + ' ') or action_str.startswith(player_marker):
                        try:
                            # Extract current state
                            current_state = extract_current_state(hand, player_idx, action_history[:-1])
                            
                            decisions.append({
                                'hand_id': hand.hand_id,
                                'action': parsed['action_type'],
                                'current_state': current_state,
                                'action_history': action_history[:-1].copy(),  # Exclude current action
                                'raw_action': action_str
                            })
                        except Exception as e:
                            # Skip if state extraction fails
                            print(f"Warning: Failed to extract state for hand {hand.hand_id}: {e}")
                            continue
            except Exception as e:
                # Skip malformed actions
                continue
    
    return decisions

def predict_optimal_action(model, feature_encoder, decision: Dict[str, Any]) -> Dict[str, Any]:
    """Predict optimal action for a decision point"""
    # Encode features
    current_features, history_sequence = feature_encoder.encode_full_state({
        'state': {
            'current_state': decision['current_state'],
            'action_history': decision['action_history']
        }
    })
    
    # Convert to tensors
    current_tensor = torch.FloatTensor([current_features])
    history_tensor = torch.FloatTensor([history_sequence])
    
    # Predict
    with torch.no_grad():
        logits = model(current_tensor, history_tensor)
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    
    # Map to action names
    action_map = {0: 'fold', 1: 'call', 2: 'raise'}
    optimal_action = action_map[predicted_class]
    
    return {
        'optimal_action': optimal_action,
        'confidence': float(confidence),
        'probabilities': {
            'fold': float(probs[0][0].item()),
            'call': float(probs[0][1].item()),
            'raise': float(probs[0][2].item())
        }
    }

def analyze_player_session(file_path: str, player_name: str) -> Dict[str, Any]:
    """Analyze a player's session and return metrics"""
    print(f"Loading model...")
    model, metadata = load_trained_model()
    feature_encoder = FeatureEncoder()
    
    print(f"Parsing file: {file_path}")
    hands = parse_hand_history(file_path, player_name=player_name)
    
    print(f"Extracting decisions for player: {player_name}")
    decisions = extract_player_decisions(hands, player_name)
    
    if not decisions:
        return {
            "error": f"No decisions found for player '{player_name}'",
            "total_hands": len(hands),
            "decisions_found": 0
        }
    
    print(f"Analyzing {len(decisions)} decisions...")
    
    # Analyze each decision
    results = []
    agreement_count = 0
    high_confidence_mistakes = []
    
    for decision in decisions:
        prediction = predict_optimal_action(model, feature_encoder, decision)
        
        player_action = decision['action']
        optimal_action = prediction['optimal_action']
        confidence = prediction['confidence']
        
        is_agreement = (player_action == optimal_action)
        if is_agreement:
            agreement_count += 1
        
        if not is_agreement and confidence > 0.7:
            high_confidence_mistakes.append({
                'hand_id': decision['hand_id'],
                'player_action': player_action,
                'optimal_action': optimal_action,
                'confidence': confidence
            })
        
        results.append({
            'hand_id': decision['hand_id'],
            'player_action': player_action,
            'optimal_action': optimal_action,
            'confidence': confidence,
            'probabilities': prediction['probabilities'],
            'agreement': is_agreement
        })
    
    # Calculate metrics
    agreement_rate = (agreement_count / len(decisions)) * 100 if decisions else 0
    
    # Group mistakes by type
    mistake_types = {}
    for mistake in high_confidence_mistakes:
        key = f"{mistake['player_action']}_to_{mistake['optimal_action']}"
        mistake_types[key] = mistake_types.get(key, 0) + 1
    
    return {
        "player_name": player_name,
        "total_hands": len(hands),
        "total_decisions": len(decisions),
        "agreement_rate": round(agreement_rate, 2),
        "agreement_count": agreement_count,
        "high_confidence_mistakes": len(high_confidence_mistakes),
        "mistake_types": mistake_types,
        "top_mistakes": sorted(mistake_types.items(), key=lambda x: x[1], reverse=True)[:5],
        "decisions": results[:50]  # Return first 50 for display
    }

if __name__ == "__main__":
    # Test with a sample file
    test_file = "../../Poker_Data_Set/data/handhq/ABS-2009-07-01_2009-07-23_50NLH_OBFU/0.5/abs NLH handhq_1-OBFUSCATED.phhs"
    results = analyze_player_session(test_file, "Hero")
    print(json.dumps(results, indent=2))

