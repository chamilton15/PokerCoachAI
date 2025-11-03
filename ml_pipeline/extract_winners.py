#!/usr/bin/env python3
"""
Step 1: Identify Top 20% Winners from handhq Dataset
Calculate BB/hand for all players and select top percentile
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import multiprocessing

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_coach.parser import HandHistoryParser


def find_all_phhs_files(directory):
    """Find all .phhs files recursively"""
    phhs_files = []
    directory = Path(directory)
    
    for file_path in directory.rglob("*.phhs"):
        phhs_files.append(file_path)
    
    return phhs_files


def _default_player_stats():
    """Helper function for defaultdict (must be top-level for pickling)"""
    return {
        'total_winnings': 0.0,
        'hands_played': 0,
        'big_blind': 0.50
    }

def calculate_hand_winnings(hand, player_idx):
    """
    Calculate net profit for a player in a hand from actions
    Uses winnings field when available (which contains gross winnings),
    otherwise infers from actions
    """
    player_marker = f'p{player_idx + 1}'
    
    # Calculate total pot and investments
    num_players = len(hand.players)
    investments = [0.0] * num_players
    
    # Add blinds/antes to investments
    for i in range(num_players):
        if i < len(hand.blinds):
            investments[i] += hand.blinds[i]
        if i < len(hand.antes):
            investments[i] += hand.antes[i]
    
    # Track current bet amounts for each player (to calculate calls)
    current_bets = [0.0] * num_players
    
    # Track investments from actions
    for action in hand.actions:
        if action.startswith('d '):  # Dealing action, skip
            continue
            
        parts = action.split()
        if len(parts) < 2:
            continue
        
        # Extract player index from action
        player_marker_in_action = parts[0]
        if not player_marker_in_action.startswith('p') or not player_marker_in_action[1:].isdigit():
            continue
        
        action_player_idx = int(player_marker_in_action[1:]) - 1
        if action_player_idx < 0 or action_player_idx >= num_players:
            continue
        
        action_type = parts[1]
        amount = 0.0
        
        if len(parts) >= 3:
            try:
                amount = float(parts[2])
            except (ValueError, IndexError):
                pass
        
        if action_type == 'cbr':  # Bet/raise
            # Amount is total to put in pot
            to_call = amount - current_bets[action_player_idx]
            if to_call > 0:
                investments[action_player_idx] += to_call
                current_bets[action_player_idx] = amount
                # Update current_bets for other players if this is a raise
                max_bet = max(current_bets)
                if amount > max_bet:
                    # This is a raise, others need to match
                    pass
        elif action_type == 'cc':  # Call
            # Call amount = current max bet - player's current bet
            if amount > 0:
                investments[action_player_idx] += amount
                current_bets[action_player_idx] += amount
            else:
                # Infer call amount from pot
                max_bet = max(current_bets)
                to_call = max_bet - current_bets[action_player_idx]
                if to_call > 0:
                    investments[action_player_idx] += to_call
                    current_bets[action_player_idx] = max_bet
        elif action_type == 'f':  # Fold - already invested, no return
            pass
    
    # Check if winnings field has actual data (not all zeros)
    if player_idx < len(hand.winnings):
        gross_winnings_from_field = hand.winnings[player_idx]
        
        # If winnings field has data for this player, use it
        # Winnings field contains GROSS winnings (what they won from pot)
        # We need to subtract their investment to get NET profit
        if gross_winnings_from_field > 0:
            # We have actual winnings data - calculate net profit
            net_profit = gross_winnings_from_field - investments[player_idx]
            return net_profit
    
    # If winnings field is all zeros or missing, fall back to action-based calculation
    # This happens when the winnings field wasn't populated in the data
    
    # Calculate total pot
    total_pot = sum(investments)
    
    # Determine winner: player who didn't fold and went to showdown
    folded_players = set()
    for action in hand.actions:
        if ' f' in action:
            parts = action.split()
            if len(parts) >= 1:
                player_marker_in_action = parts[0]
                if player_marker_in_action.startswith('p') and player_marker_in_action[1:].isdigit():
                    folded_idx = int(player_marker_in_action[1:]) - 1
                    if 0 <= folded_idx < num_players:
                        folded_players.add(folded_idx)
    
    # Check if this player folded
    player_folded = player_idx in folded_players
    
    # If player didn't fold, they might have won
    if not player_folded:
        # Count non-folding players
        non_folding_count = num_players - len(folded_players)
        
        if non_folding_count == 1:
            # This player won the entire pot
            won = total_pot
        elif non_folding_count > 1:
            # Multiple players at showdown - split pot (approximate)
            # This is a simplification - actual pot distribution would require hand evaluation
            won = total_pot / non_folding_count
        else:
            won = 0.0
    else:
        won = 0.0
    
    # Net profit = winnings - investment
    net_profit = won - investments[player_idx]
    return net_profit


def process_single_file(file_path):
    """Process a single .phhs file and return player stats"""
    file_stats = defaultdict(_default_player_stats)
    hands_count = 0
    
    try:
        parser = HandHistoryParser(str(file_path))
        hands = parser.parse()
        
        for hand in hands:
            hands_count += 1
            
            # Get big blind from hand
            if len(hand.blinds) >= 2:
                big_blind = hand.blinds[1]
            else:
                big_blind = 0.50  # Default
            
            # Process each player in the hand
            for player_idx, player_name in enumerate(hand.players):
                # Calculate winnings from actions since winnings field is often 0
                winnings = calculate_hand_winnings(hand, player_idx)
                
                # Update stats
                file_stats[player_name]['total_winnings'] += winnings
                file_stats[player_name]['hands_played'] += 1
                
                # Update big blind (use most common)
                if file_stats[player_name]['big_blind'] == 0.50:
                    file_stats[player_name]['big_blind'] = big_blind
    
    except Exception as e:
        return file_stats, hands_count, str(e)
    
    return file_stats, hands_count, None


def calculate_player_stats(phhs_files, min_hands=20, num_workers=None):
    """
    Parse all handhq files and calculate statistics for each player
    Uses parallel processing for speed.
    
    Returns:
        dict: {player_name: {'total_winnings': float, 'hands_played': int, 'bb_per_hand': float}}
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one core free
    
    print(f"Parsing {len(phhs_files)} files using {num_workers} workers...")
    
    # Aggregate stats from all files
    player_stats = defaultdict(_default_player_stats)
    
    total_hands_parsed = 0
    errors = []
    
    # Set start method for macOS compatibility (spawn is default on macOS)
    # We need to use 'spawn' method which requires the __main__ guard
    ctx = multiprocessing.get_context('spawn')
    
    # Process files in parallel with real-time progress updates
    with ctx.Pool(processes=num_workers) as pool:
        # Use imap_unordered for real-time progress
        results_iter = pool.imap_unordered(process_single_file, phhs_files, chunksize=10)
        
        results = []
        completed = 0
        
        for result in results_iter:
            results.append(result)
            completed += 1
            
            if completed % 50 == 0 or completed == len(phhs_files):
                print(f"  Processed {completed}/{len(phhs_files)} files...")
        
    # Aggregate results (order doesn't matter for aggregation)
    for file_stats, hands_count, error in results:
        if error:
            errors.append(('unknown', error))  # File path lost with imap_unordered
        
        total_hands_parsed += hands_count
        
        # Merge file stats into global stats
        for player_name, stats in file_stats.items():
            player_stats[player_name]['total_winnings'] += stats['total_winnings']
            player_stats[player_name]['hands_played'] += stats['hands_played']
            
            # Update big blind (use most common)
            if player_stats[player_name]['big_blind'] == 0.50 and stats['big_blind'] != 0.50:
                player_stats[player_name]['big_blind'] = stats['big_blind']
    
    if errors:
        print(f"\n  Warnings: {len(errors)} files had errors (first 5):")
        for file_path, error in errors[:5]:
            if isinstance(file_path, Path):
                print(f"    {file_path.name}: {error}")
            else:
                print(f"    {file_path}: {error}")
    
    print(f"\nParsed {total_hands_parsed} total hands")
    print(f"Found {len(player_stats)} unique players")
    
    # Calculate BB/hand for valid players
    valid_players = {}
    for player_name, stats in player_stats.items():
        if stats['hands_played'] >= min_hands:
            total_winnings = stats['total_winnings']
            num_hands = stats['hands_played']
            big_blind = stats['big_blind']
            
            # Calculate BB/hand
            if big_blind > 0:
                bb_per_hand = (total_winnings / num_hands) / big_blind
            else:
                bb_per_hand = 0.0
            
            valid_players[player_name] = {
                'bb_per_hand': bb_per_hand,
                'hands_played': num_hands,
                'total_winnings': total_winnings,
                'big_blind': big_blind
            }
    
    print(f"Valid players (≥{min_hands} hands): {len(valid_players)}")
    
    return valid_players


def identify_top_winners(valid_players, top_percentile=20):
    """
    Identify top percentile winners
    
    Args:
        valid_players: Dict of player stats
        top_percentile: Top X% to select (default 20)
    
    Returns:
        list: Player names in top percentile
        dict: Full stats for winners
    """
    # Sort players by BB/hand (descending)
    sorted_players = sorted(
        valid_players.items(),
        key=lambda x: x[1]['bb_per_hand'],
        reverse=True
    )
    
    # Calculate threshold
    num_players = len(sorted_players)
    top_count = max(1, int(num_players * (top_percentile / 100)))
    
    # Extract winners
    winners = [player_name for player_name, _ in sorted_players[:top_count]]
    winner_stats = {name: valid_players[name] for name in winners}
    
    # Print statistics
    if sorted_players:
        winner_bb_values = [stats['bb_per_hand'] for _, stats in sorted_players[:top_count]]
        all_bb_values = [stats['bb_per_hand'] for _, stats in sorted_players]
        
        # Calculate statistics manually (no numpy dependency)
        def calculate_stats(values):
            if not values:
                return 0.0, 0.0
            sorted_vals = sorted(values)
            median = sorted_vals[len(sorted_vals) // 2] if len(sorted_vals) > 0 else 0.0
            mean = sum(values) / len(values) if len(values) > 0 else 0.0
            return median, mean
        
        all_median, all_mean = calculate_stats(all_bb_values)
        winner_median, winner_mean = calculate_stats(winner_bb_values)
        
        print(f"\n{'='*70}")
        print(f"WINNER IDENTIFICATION RESULTS")
        print(f"{'='*70}")
        print(f"Total valid players: {num_players}")
        print(f"Winners selected (top {top_percentile}%): {len(winners)}")
        print(f"\nBB/hand Statistics:")
        print(f"  All players - Median: {all_median:.3f}, Mean: {all_mean:.3f}")
        print(f"  Winners - Median: {winner_median:.3f}, Mean: {winner_mean:.3f}")
        print(f"  Winners - Range: {max(winner_bb_values):.3f} to {min(winner_bb_values):.3f}")
        print(f"\nTotal hands from winners: {sum(s['hands_played'] for s in winner_stats.values())}")
        
        # Top 5 winners
        print(f"\nTop 5 Winners:")
        for i, (name, stats) in enumerate(sorted_players[:5], 1):
            print(f"  {i}. {name[:30]:30s} - {stats['bb_per_hand']:6.3f} BB/hand "
                  f"({stats['hands_played']} hands)")
    
    return winners, winner_stats


def main():
    """Main execution"""
    
    # Configuration
    handhq_directory = "/Users/sethfgn/Desktop/DL_Poker_Project/Poker_Data_Set/data/handhq"
    output_file = "winners_list.json"
    min_hands = 20  # Minimum hands to be considered
    top_percentile = 20  # Top 20%
    num_workers = None  # Auto-detect (uses cpu_count() - 1)
    
    print("="*70)
    print("STEP 1: IDENTIFYING TOP WINNERS FROM HANDHQ DATASET")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Directory: {handhq_directory}")
    print(f"  Minimum hands: {min_hands}")
    print(f"  Top percentile: {top_percentile}%")
    print()
    
    # Find all .phhs files
    print("Finding .phhs files...")
    phhs_files = find_all_phhs_files(handhq_directory)
    print(f"Found {len(phhs_files)} .phhs files")
    
    if len(phhs_files) == 0:
        print("ERROR: No .phhs files found!")
        return
    
    # Calculate player statistics (with parallel processing)
    valid_players = calculate_player_stats(phhs_files, min_hands=min_hands, num_workers=num_workers)
    
    if len(valid_players) == 0:
        print("ERROR: No valid players found!")
        return
    
    # Identify winners
    winners, winner_stats = identify_top_winners(valid_players, top_percentile=top_percentile)
    
    # Save results
    output_data = {
        'winners': winners,
        'winner_stats': winner_stats,
        'total_valid_players': len(valid_players),
        'top_percentile': top_percentile,
        'min_hands': min_hands
    }
    
    output_path = Path(__file__).parent / output_file
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")
    
    return winners, winner_stats


if __name__ == "__main__":
    winners, stats = main()

