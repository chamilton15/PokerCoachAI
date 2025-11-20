#!/usr/bin/env python3
"""Verify top winners have legitimate high BB/hand ratios"""

import json
from pathlib import Path

# Load winners data
with open('winners_list.json', 'r') as f:
    data = json.load(f)

winners = data['winners']
winner_stats = data['winner_stats']

print("="*90)
print("VERIFICATION: Top 10 Winners Analysis")
print("="*90)

# Get top 10 winners
top_10 = winners[:10]

for i, name in enumerate(top_10, 1):
    stats = winner_stats[name]
    
    print(f"\n{i}. {name[:50]}")
    print(f"   {'-'*80}")
    
    # Extract values
    bb_per_hand = stats['bb_per_hand']
    total_winnings = stats['total_winnings']
    hands_played = stats['hands_played']
    big_blind = stats['big_blind']
    
    # Manual calculation check
    if big_blind > 0:
        manual_calc = (total_winnings / hands_played) / big_blind
    else:
        manual_calc = 0.0
    
    # Display stats
    print(f"   BB/hand:           {bb_per_hand:10.4f}")
    print(f"   Total winnings:    ${total_winnings:12.2f}")
    print(f"   Hands played:      {hands_played:10d}")
    print(f"   Big blind:         ${big_blind:12.2f}")
    print(f"   Winnings/hand:     ${total_winnings/hands_played:12.4f}")
    
    # Verification
    calc_match = abs(manual_calc - bb_per_hand) < 0.001
    print(f"   Manual calc check: {manual_calc:10.4f} {'✓' if calc_match else '✗ MATCH FAILED'}")
    
    # Sanity checks
    issues = []
    if bb_per_hand < 0:
        issues.append("Negative BB/hand!")
    if hands_played < 20:
        issues.append(f"Low hand count: {hands_played}")
    if big_blind <= 0:
        issues.append(f"Invalid big blind: {big_blind}")
    if abs(total_winnings) > 1000000:
        issues.append(f"Very large winnings: ${total_winnings:.2f}")
    
    if issues:
        print(f"   ⚠ WARNINGS: {'; '.join(issues)}")
    else:
        print(f"   ✓ All checks passed")

# Overall statistics
print(f"\n{'='*90}")
print("OVERALL STATISTICS")
print(f"{'='*90}")

all_bb_values = [s['bb_per_hand'] for s in winner_stats.values()]
sorted_bb = sorted(all_bb_values)

print(f"Total winners: {len(winners)}")
print(f"BB/hand range: {min(all_bb_values):.3f} to {max(all_bb_values):.3f}")
print(f"Median BB/hand: {sorted_bb[len(sorted_bb)//2]:.3f}")
print(f"Mean BB/hand: {sum(all_bb_values)/len(all_bb_values):.3f}")

# Check top 10 against overall stats
top10_bb = [winner_stats[w]['bb_per_hand'] for w in top_10]
print(f"\nTop 10 BB/hand values: {[round(x, 3) for x in top10_bb]}")
print(f"Top 10 vs Median: {sorted_bb[len(sorted_bb)//2]:.3f} (top 10 are {min(top10_bb)/sorted_bb[len(sorted_bb)//2]:.1f}x - {max(top10_bb)/sorted_bb[len(sorted_bb)//2]:.1f}x higher)")

print(f"\n{'='*90}")
print("VERIFICATION COMPLETE")
print(f"{'='*90}")


