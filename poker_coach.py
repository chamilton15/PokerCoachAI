#!/usr/bin/env python3
"""
Poker Coach AI - Main script
Analyze poker hand histories and provide coaching feedback
"""

import sys
import os
from pathlib import Path

# Add poker_coach to path
sys.path.insert(0, str(Path(__file__).parent))

from poker_coach.parser import parse_hand_history
from poker_coach.statistics import PokerStatistics
from poker_coach.analyzer import PlayerAnalyzer
from poker_coach.feedback import FeedbackGenerator


def analyze_player(file_path: str, player_name: str = "Hero") -> str:
    """
    Analyze a player's poker session
    
    Args:
        file_path: Path to .phhs file
        player_name: Name of player to analyze (default: "Hero")
    
    Returns:
        Coaching report as string
    """
    
    print(f"Loading hand history: {file_path}")
    print(f"Analyzing player: {player_name}")
    print("=" * 70)
    
    # Parse hand history
    hands = parse_hand_history(file_path, player_name=player_name)
    
    if not hands:
        return f"Error: No hands found for player '{player_name}' in {file_path}"
    
    print(f"Loaded {len(hands)} hands for {player_name}")
    
    # Calculate statistics
    print("Calculating statistics...")
    stats_calculator = PokerStatistics(hands, player_name)
    stats = stats_calculator.calculate_all()
    
    # Analyze strategy
    print("Analyzing strategy vs baseline...")
    analyzer = PlayerAnalyzer(hands, player_name)
    mistakes, summary = analyzer.analyze_all_hands()
    patterns = analyzer.get_patterns()
    
    print(f"Found {len(mistakes)} strategic deviations")
    
    # Generate report
    print("Generating coaching report...")
    print("=" * 70)
    print()
    
    report = FeedbackGenerator.generate_report(
        player_name=player_name,
        stats=stats,
        mistakes=mistakes,
        patterns=patterns,
        num_hands=len(hands)
    )
    
    return report


def main():
    """Main entry point"""
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python poker_coach.py <hand_history_file> [player_name]")
        print()
        print("Examples:")
        print("  python poker_coach.py abs_NLH_handhq_1_Hero_extracted.phhs")
        print("  python poker_coach.py my_session.phhs PlayerName")
        print()
        sys.exit(1)
    
    file_path = sys.argv[1]
    player_name = sys.argv[2] if len(sys.argv) > 2 else "Hero"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Run analysis
    report = analyze_player(file_path, player_name)
    print(report)
    
    # Save report to file
    output_file = f"{player_name}_analysis_report.txt"
    with open(output_file, 'w') as f:
        f.write(report)
    
    print()
    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    main()

