#!/usr/bin/env python3
"""
Step 2: Extract Training Data with Full Action History
For each winner's decision, extract:
- Current game state
- Complete action history up to that point
- Label: What the winner actually did
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import multiprocessing

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_coach.parser import HandHistoryParser, Hand
from poker_coach.hand_strength import HandStrengthEvaluator


def load_winners(winners_file="winners_list.json"):
    """Load winner list from Step 1"""
    winners_path = Path(__file__).parent / winners_file
    
    if not winners_path.exists():
        print(f"ERROR: {winners_file} not found. Run extract_winners.py first!")
        return None, None
    
    with open(winners_path, 'r') as f:
        data = json.load(f)
    
    return data['winners'], data['winner_stats']


def parse_action_string(action_str: str) -> Dict[str, Any]:
    """Parse an action string into structured format"""
    parts = action_str.split()
    
    if len(parts) < 2:
        return None
    
    # Extract player marker (p1, p2, etc.)
    player_marker = parts[0]
    player_idx = int(player_marker[1:]) - 1  # Convert p1 -> 0, p2 -> 1
    
    # Extract action type
    action_type = parts[1]  # 'f', 'cc', 'cbr', 'cr'
    
    # Extract amount if present
    amount = None
    if len(parts) > 2:
        try:
            amount = float(parts[2])
        except ValueError:
            amount = None
    
    # Normalize action types
    action_map = {
        'f': 'fold',
        'cc': 'call',
        'cbr': 'raise',
        'cr': 'raise',
        'sd': 'showdown',
        'sm': 'showdown'
    }
    
    normalized_action = action_map.get(action_type, action_type)
    
    return {
        'player_idx': player_idx,
        'action_type': normalized_action,
        'raw_action': action_type,
        'amount': amount,
        'raw_string': action_str
    }


def extract_action_history(hand: Hand, up_to_index: int) -> List[Dict[str, Any]]:
    """
    Extract action history up to a specific point in the hand
    
    Args:
        hand: Hand object
        up_to_index: Extract actions up to this index (exclusive)
    
    Returns:
        List of action dictionaries with full context
    """
    history = []
    street = 'preflop'
    pot_size = sum(hand.blinds[:2]) if len(hand.blinds) >= 2 else 0.75
    
    for i, action_str in enumerate(hand.actions[:up_to_index]):
        # Check for street changes
        if action_str.startswith('d db'):
            # Board cards dealt
            if street == 'preflop':
                street = 'flop'
            elif street == 'flop':
                street = 'turn'
            elif street == 'turn':
                street = 'river'
        
        # Parse player actions
        if action_str.startswith('p'):
            parsed = parse_action_string(action_str)
            if parsed:
                parsed['street'] = street
                parsed['sequence_index'] = len(history)
                parsed['pot_size_before'] = pot_size
                
                # Update pot size
                if parsed['action_type'] == 'raise' and parsed['amount']:
                    pot_size += parsed['amount']
                elif parsed['action_type'] == 'call':
                    # Estimate call amount (would need more context in real implementation)
                    pass
                
                parsed['pot_size_after'] = pot_size
                history.append(parsed)
    
    return history


def determine_facing_action(action_history: List[Dict], current_player_idx: int) -> str:
    """Determine what action the current player is facing"""
    if not action_history:
        return 'none'
    
    # Look backwards for last action before current player
    facing = 'none'
    raises_count = 0
    
    for action in reversed(action_history):
        if action['player_idx'] != current_player_idx:
            if action['action_type'] == 'raise':
                raises_count += 1
                if raises_count == 1:
                    facing = 'raise'
                elif raises_count == 2:
                    facing = '3bet'
                    break
            elif action['action_type'] in ['call', 'fold']:
                if facing == 'none':
                    facing = 'none'  # No raise yet
    
    return facing


def extract_current_state(hand: Hand, player_idx: int, history: List[Dict]) -> Dict[str, Any]:
    """Extract current game state at decision point"""
    
    # Get player's hole cards
    player_marker = f'p{player_idx + 1}'
    hole_cards = None
    hand_score = 5  # Default
    hand_type = 'unknown'
    
    evaluator = HandStrengthEvaluator()
    for action in hand.actions:
        if action.startswith(f'd dh {player_marker}'):
            cards_str = action.split()[-1]
            cards = evaluator.parse_cards(cards_str)
            if cards:
                hole_cards = cards
                hand_score, hand_type = evaluator.evaluate_preflop_strength(cards)
            break
    
    # Determine street
    street = 'preflop'
    board_cards = []
    for action in hand.actions:
        if action.startswith('d db'):
            board_cards_str = action.split()[2:]
            board_cards.extend([c for c in board_cards_str if len(c) == 2])
            if len(board_cards) == 3:
                street = 'flop'
            elif len(board_cards) == 4:
                street = 'turn'
            elif len(board_cards) == 5:
                street = 'river'
    
            # Get position
            if player_idx < len(hand.players):
                position = hand.get_player_position(hand.players[player_idx])
            else:
                position = 'unknown'
    
    # Calculate pot size
    pot_size = sum(hand.blinds[:2]) if len(hand.blinds) >= 2 else 0.75
    for action in history:
        if action.get('amount'):
            pot_size += action['amount']
    
    # Get stack size (use starting stack as approximation)
    stack_size = hand.starting_stacks[player_idx] if player_idx < len(hand.starting_stacks) else 50.0
    
    return {
        'hole_cards': hole_cards,
        'hand_score': hand_score,
        'hand_type': hand_type,
        'position': position or 'unknown',
        'street': street,
        'board_cards': board_cards,
        'pot_size': pot_size,
        'stack_size': stack_size,
        'num_opponents': len(hand.players) - 1,
        'facing_action': determine_facing_action(history, player_idx)
    }


def extract_decision_with_history(hand: Hand, winner_player_idx: int) -> List[Dict[str, Any]]:
    """
    Extract all decision points for a winner in a hand with full action history
    
    IMPORTANT: action_history includes ALL players' actions, not just the winner's!
    This gives the model full context of what happened in the hand.
    
    Returns:
        List of training examples: (state_with_history, label)
    """
    decisions = []
    player_marker = f'p{winner_player_idx + 1}'
    
    # Track action history as we process - includes ALL players' actions
    action_history = []
    street = 'preflop'
    pot_size = sum(hand.blinds[:2]) if len(hand.blinds) >= 2 else 0.75
    
    for i, action_str in enumerate(hand.actions):
        # Check for street changes (affects all subsequent actions)
        if action_str.startswith('d db'):
            # Board cards dealt
            if street == 'preflop':
                street = 'flop'
            elif street == 'flop':
                street = 'turn'
            elif street == 'turn':
                street = 'river'
            continue  # Skip dealing actions, they're not player actions
        
        # Parse ALL player actions (not just winner's)
        if action_str.startswith('p'):
            parsed = parse_action_string(action_str)
            if parsed:
                # Add street and pot context to ALL actions
                parsed['street'] = street
                parsed['pot_size_before'] = pot_size
                
                # Update pot size estimate
                if parsed['action_type'] == 'raise' and parsed['amount']:
                    pot_size += parsed['amount']
                elif parsed['action_type'] == 'call' and parsed['amount']:
                    pot_size += parsed['amount']
                
                parsed['pot_size_after'] = pot_size
                
                # Add ALL actions to history (this is the key fix!)
                action_history.append(parsed)
                
                # Only extract decision point when it's the winner's turn
                if action_str.startswith(player_marker):
                    # This is winner's decision point!
                    
                    # Extract current state (history includes all players' actions up to now)
                    current_state = extract_current_state(hand, winner_player_idx, action_history)
                    
                    # Build full state with history
                    # Note: action_history now contains ALL players' actions, not just winner's
                    full_state = {
                        'current_state': current_state,
                        'action_history': action_history[:-1].copy(),  # Exclude winner's current action
                        'history_length': len(action_history) - 1,
                        
                        # Derived features from history
                        'num_raises_in_history': sum(1 for a in action_history[:-1] if a['action_type'] == 'raise'),
                        'num_calls_in_history': sum(1 for a in action_history[:-1] if a['action_type'] == 'call'),
                        'num_folds_in_history': sum(1 for a in action_history[:-1] if a['action_type'] == 'fold'),
                        'last_action': action_history[-2] if len(action_history) > 1 else None,
                    }
                    
                    # Create training example
                    training_example = {
                        'hand_id': hand.hand_id,
                        'state': full_state,
                        'label': parsed['action_type'],  # What winner actually did
                        'raw_action': parsed['raw_action'],
                        'sequence_position': len(action_history) - 1,
                    }
                    
                    decisions.append(training_example)
    
    return decisions


def _default_int():
    """Helper function for defaultdict (must be top-level for pickling)"""
    return 0

def process_file_for_winners(args):
    """Process a single file for winner training data extraction"""
    file_path, winners = args
    
    file_training_data = []
    file_winner_hands_count = defaultdict(_default_int)
    
    try:
        parser = HandHistoryParser(str(file_path))
        hands = parser.parse()
        
        for hand in hands:
            # Check if any winner is in this hand
            for winner in winners:
                if winner in hand.players:
                    winner_idx = hand.get_player_index(winner)
                    if winner_idx is not None:
                        # Extract decisions with history
                        decisions = extract_decision_with_history(hand, winner_idx)
                        
                        for decision in decisions:
                            decision['winner'] = winner
                            file_training_data.append(decision)
                        
                        file_winner_hands_count[winner] += 1
                        break  # Only count hand once per winner
    
    except Exception as e:
        return file_training_data, file_winner_hands_count, str(e)
    
    return file_training_data, file_winner_hands_count, None


def process_winner_hands(winners: List[str], handhq_directory: str, num_workers=None) -> List[Dict[str, Any]]:
    """Process all hands for winners and extract training data (with parallel processing)"""
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one core free
    
    all_training_data = []
    handhq_path = Path(handhq_directory)
    
    print(f"\nProcessing hands for {len(winners)} winners...")
    
    # Find all .phhs files
    phhs_files = list(handhq_path.rglob("*.phhs"))
    print(f"Found {len(phhs_files)} .phhs files to process")
    print(f"Using {num_workers} workers for parallel processing")
    
    # Aggregate winner hands count
    winner_hands_count = defaultdict(int)
    errors = []
    
    # Prepare arguments for parallel processing
    file_args = [(file_path, winners) for file_path in phhs_files]
    
    # Set start method for macOS compatibility (spawn is default on macOS)
    ctx = multiprocessing.get_context('spawn')
    
    # Process files in parallel with real-time progress updates
    with ctx.Pool(processes=num_workers) as pool:
        # Use imap_unordered for real-time progress
        results_iter = pool.imap_unordered(process_file_for_winners, file_args, chunksize=10)
        
        results = []
        completed = 0
        
        for result in results_iter:
            results.append(result)
            completed += 1
            
            # Aggregate immediately for progress reporting
            file_data, file_counts, error = result
            if error:
                errors.append((None, error))  # File path lost with imap_unordered
            
            all_training_data.extend(file_data)
            
            # Merge counts
            for winner, count in file_counts.items():
                winner_hands_count[winner] += count
            
            # Real-time progress updates
            if completed % 50 == 0 or completed == len(phhs_files):
                print(f"  Processed {completed}/{len(phhs_files)} files... "
                      f"(Found {len(all_training_data)} training examples so far)")
    
    if errors:
        print(f"\n  Warnings: {len(errors)} files had errors (showing first 5):")
        for file_path, error in errors[:5]:
            if isinstance(file_path, Path):
                print(f"    {file_path.name}: {error}")
            else:
                print(f"    {file_path or 'unknown'}: {error}")
    
    print(f"\n{'='*70}")
    print(f"TRAINING DATA EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total training examples: {len(all_training_data)}")
    print(f"\nHands processed per winner:")
    for winner, count in sorted(winner_hands_count.items(), key=lambda x: -x[1])[:10]:
        print(f"  {winner[:40]:40s}: {count:4d} hands")
    
    # Statistics
    labels = [d['label'] for d in all_training_data]
    from collections import Counter
    label_counts = Counter(labels)
    
    print(f"\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label:10s}: {count:6d} ({count/len(all_training_data)*100:5.1f}%)")
    
    return all_training_data


def main():
    """Main execution"""
    
    # Configuration
    handhq_directory = "/Users/sethfgn/Desktop/DL_Poker_Project/Poker_Data_Set/data/handhq"
    winners_file = "winners_list.json"
    output_file = "training_data.json"
    num_workers = None  # Auto-detect (uses cpu_count() - 1)
    
    print("="*70)
    print("STEP 2: EXTRACTING TRAINING DATA WITH ACTION HISTORY")
    print("="*70)
    
    # Load winners from Step 1
    winners, winner_stats = load_winners(winners_file)
    if winners is None:
        return
    
    print(f"\nLoaded {len(winners)} winners from Step 1")
    
    # Process all winner hands (with parallel processing)
    training_data = process_winner_hands(winners, handhq_directory, num_workers=num_workers)
    
    if len(training_data) == 0:
        print("ERROR: No training data extracted!")
        return
    
    # Save training data
    output_path = Path(__file__).parent / output_file
    
    # Save in chunks if too large (for memory efficiency)
    print(f"\nSaving training data to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump({
            'training_data': training_data,
            'num_examples': len(training_data),
            'winners': winners,
            'metadata': {
                'total_winners': len(winners),
                'handhq_directory': handhq_directory
            }
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Training data saved to: {output_path}")
    print(f"Total examples: {len(training_data)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

