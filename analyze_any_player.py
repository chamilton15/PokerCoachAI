#!/usr/bin/env python3
"""
Extract and analyze any player from any .phhs file
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from poker_coach.parser import HandHistoryParser
from poker_coach.statistics import PokerStatistics
from poker_coach.analyzer import PlayerAnalyzer
from poker_coach.feedback import FeedbackGenerator


def extract_and_analyze(source_file: str, player_id: str, output_name: str = None):
    """
    Extract hands for a specific player and analyze them
    
    Args:
        source_file: Path to source .phhs file
        player_id: Player ID to extract (can be partial match)
        output_name: Optional friendly name for the player
    """
    
    if not os.path.exists(source_file):
        print(f"Error: File not found: {source_file}")
        return
    
    print(f"Loading: {source_file}")
    print(f"Looking for player: {player_id}")
    print("=" * 70)
    
    # Parse all hands
    parser = HandHistoryParser(source_file)
    all_hands = parser.parse()
    
    print(f"Total hands in file: {len(all_hands)}")
    
    # Find player hands
    player_hands = []
    player_full_name = None
    
    for hand in all_hands:
        # Check if player_id matches any player in this hand
        for player in hand.players:
            if player_id in player or player in player_id:
                player_hands.append(hand)
                if not player_full_name:
                    player_full_name = player
                break
    
    if not player_hands:
        print(f"\nError: No hands found for player matching '{player_id}'")
        print("\nPlayers found in first hand:")
        if all_hands:
            for p in all_hands[0].players:
                print(f"  - {p}")
        return
    
    print(f"Found {len(player_hands)} hands for: {player_full_name}")
    
    if len(player_hands) < 20:
        print(f"\nWarning: Only {len(player_hands)} hands found. Analysis works best with 30+ hands.")
    
    # Use friendly name if provided
    analysis_name = output_name if output_name else player_full_name
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats_calc = PokerStatistics(player_hands, player_full_name)
    stats = stats_calc.calculate_all()
    
    # Analyze strategy
    print("Analyzing strategy...")
    analyzer = PlayerAnalyzer(player_hands, player_full_name)
    mistakes, summary = analyzer.analyze_all_hands()
    patterns = analyzer.get_patterns()
    
    print(f"Found {len(mistakes)} strategic deviations\n")
    
    # Generate report
    report = FeedbackGenerator.generate_report(
        player_name=analysis_name,
        stats=stats,
        mistakes=mistakes,
        patterns=patterns,
        num_hands=len(player_hands)
    )
    
    print(report)
    
    # Save report
    safe_name = analysis_name.replace('/', '_').replace(' ', '_')
    output_file = f"{safe_name}_analysis_report.txt"
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\n\nReport saved to: {output_file}")
    
    # Also save extracted hands
    hands_file = f"{safe_name}_extracted_hands.phhs"
    with open(hands_file, 'w') as f:
        for hand in player_hands:
            f.write(f"[{hand.hand_id}]\n")
            f.write(f"variant = '{hand.variant}'\n")
            f.write(f"antes = {hand.antes}\n")
            f.write(f"blinds_or_straddles = {hand.blinds}\n")
            f.write(f"min_bet = {hand.min_bet}\n")
            f.write(f"starting_stacks = {hand.starting_stacks}\n")
            f.write(f"actions = {hand.actions}\n")
            f.write(f"venue = '{hand.venue}'\n")
            f.write(f"time = {hand.time}\n")
            f.write(f"day = {hand.date['day']}\n")
            f.write(f"month = {hand.date['month']}\n")
            f.write(f"year = {hand.date['year']}\n")
            f.write(f"hand = {hand.hand_number}\n")
            f.write(f"seats = {hand.seats}\n")
            f.write(f"table = '{hand.table}'\n")
            f.write(f"players = {hand.players}\n")
            f.write(f"winnings = {hand.winnings}\n")
            f.write(f"currency_symbol = '{hand.currency_symbol}'\n")
            f.write("\n")
    
    print(f"Extracted hands saved to: {hands_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_any_player.py <phhs_file> <player_id> [output_name]")
        print()
        print("Examples:")
        print("  python analyze_any_player.py data.phhs l1bCLGuFqeFRwUfPsiDu/g Player1")
        print("  python analyze_any_player.py data.phhs Hero")
        print()
        sys.exit(1)
    
    source_file = sys.argv[1]
    player_id = sys.argv[2]
    output_name = sys.argv[3] if len(sys.argv) > 3 else None
    
    extract_and_analyze(source_file, player_id, output_name)

