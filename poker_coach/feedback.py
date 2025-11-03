"""
Generate coaching feedback and recommendations
"""

from typing import List, Dict
from collections import defaultdict

from .analyzer import Mistake


class FeedbackGenerator:
    """Generate actionable coaching feedback"""
    
    MISTAKE_DESCRIPTIONS = {
        'over_folding': {
            'title': 'STOP OVER-FOLDING',
            'issue': "You're folding too often in profitable situations",
            'impact': "You're leaving money on the table by being too cautious",
            'advice': [
                "Call more often with medium-strength hands when in position",
                "Don't treat single raises as super-strong - many players raise wide",
                "Consider your pot odds - if getting 3:1 or better, call lighter",
                "Medium pairs (66-99) are often profitable calls in position"
            ]
        },
        'over_calling': {
            'title': 'REDUCE WEAK CALLS',
            'issue': "You're calling too often with weak hands",
            'impact': "Calling with weak hands in bad spots loses money over time",
            'advice': [
                "Be more selective about which hands you call with",
                "Fold more often when out of position with marginal hands",
                "Consider 3-betting or folding instead of passively calling",
                "Respect raises from tight players - they usually have it"
            ]
        },
        'under_aggressive': {
            'title': 'INCREASE AGGRESSION',
            'issue': "You're playing too passively with strong hands",
            'impact': "Passive play misses value and makes you predictable",
            'advice': [
                "Raise more often with premium hands - build the pot",
                "Don't slow-play too often - aggressive play wins more",
                "Add some 3-bets to your game with strong hands",
                "Bet for value when you think you're ahead"
            ]
        },
        'strategy_deviation': {
            'title': 'STRATEGY ADJUSTMENTS NEEDED',
            'issue': "Your play deviates from optimal strategy",
            'impact': "These deviations add up over time",
            'advice': [
                "Review fundamental poker strategy",
                "Consider position more in your decisions",
                "Balance your ranges - mix up your play",
                "Study GTO (Game Theory Optimal) basics"
            ]
        }
    }
    
    @staticmethod
    def generate_report(
        player_name: str,
        stats: Dict,
        mistakes: List[Mistake],
        patterns: Dict[str, List[Mistake]],
        num_hands: int
    ) -> str:
        """Generate full coaching report"""
        
        report = []
        report.append("=" * 70)
        report.append("POKER COACH AI - SESSION ANALYSIS")
        report.append("=" * 70)
        report.append("")
        report.append(f"PLAYER: {player_name}")
        report.append(f"HANDS ANALYZED: {num_hands}")
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS")
        report.append("=" * 70)
        report.append(f"VPIP: {stats.get('vpip', 0):.1f}% (Optimal range: 20-30%)")
        report.append(f"PFR:  {stats.get('pfr', 0):.1f}% (Optimal range: 15-22%)")
        report.append(f"Aggression Factor: {stats.get('aggression_factor', 0):.2f} (Target: 2.0-2.5)")
        report.append(f"3-Bet: {stats.get('three_bet_pct', 0):.1f}%")
        report.append(f"C-Bet: {stats.get('cbet_pct', 0):.1f}% (Optimal: 60-70%)")
        report.append("")
        
        # Position breakdown
        if stats.get('position_stats'):
            report.append("POSITION BREAKDOWN")
            report.append("=" * 70)
            for pos, pos_stats in sorted(stats['position_stats'].items()):
                report.append(f"{pos:6s}: {pos_stats['hands']:2d} hands, "
                            f"VPIP {pos_stats['vpip']:.1f}%, "
                            f"PFR {pos_stats['pfr']:.1f}%")
            report.append("")
        
        # Strategy analysis
        report.append("STRATEGY ANALYSIS")
        report.append("=" * 70)
        total_mistakes = len(mistakes)
        agreement_rate = ((num_hands - total_mistakes) / num_hands) * 100 if num_hands > 0 else 0
        
        report.append(f"Agreement with Baseline: {agreement_rate:.1f}%")
        report.append(f"Strategic Mistakes Found: {total_mistakes}")
        report.append("")
        
        if total_mistakes > 0:
            # Mistake breakdown
            report.append("MISTAKE BREAKDOWN:")
            mistake_counts = defaultdict(int)
            for m in mistakes:
                mistake_counts[m.mistake_type] += 1
            
            for mistake_type, count in sorted(mistake_counts.items(), key=lambda x: -x[1]):
                pct = (count / total_mistakes) * 100
                report.append(f"  • {mistake_type.replace('_', ' ').title()}: {count} ({pct:.0f}%)")
            report.append("")
        
        # Top recommendations
        report.append("TOP RECOMMENDATIONS")
        report.append("=" * 70)
        report.append("")
        
        recommendations = FeedbackGenerator._generate_recommendations(patterns, num_hands)
        
        for i, rec in enumerate(recommendations[:3], 1):
            report.append(f"#{i}: {rec['title']}")
            report.append("-" * 70)
            report.append(f"\nISSUE:")
            report.append(f"  {rec['issue']}")
            report.append(f"  Frequency: {rec['frequency']}")
            report.append("")
            report.append(f"EXAMPLES:")
            for example in rec['examples'][:2]:
                report.append(f"  • {example}")
            report.append("")
            report.append(f"WHY THIS MATTERS:")
            report.append(f"  {rec['impact']}")
            report.append("")
            report.append(f"HOW TO IMPROVE:")
            for advice in rec['advice']:
                report.append(f"  ✓ {advice}")
            report.append("")
            report.append(f"ESTIMATED IMPACT: {rec['estimated_impact']}")
            report.append("")
        
        # Summary
        report.append("=" * 70)
        report.append("SUMMARY")
        report.append("=" * 70)
        report.append("")
        
        # Overall assessment
        if agreement_rate >= 80:
            report.append("You're playing solid fundamental poker! Focus on the recommendations")
            report.append("above to take your game to the next level.")
        elif agreement_rate >= 65:
            report.append("You have a decent foundation but several strategic leaks to fix.")
            report.append("Work on the recommendations above to improve your win rate.")
        else:
            report.append("Significant strategic improvements needed. Focus on fundamentals:")
            report.append("position, hand selection, and aggression. Study the recommendations carefully.")
        
        report.append("")
        report.append("Keep up the good work and keep learning!")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    @staticmethod
    def _generate_recommendations(patterns: Dict[str, List[Mistake]], num_hands: int) -> List[Dict]:
        """Generate specific recommendations from mistake patterns"""
        recommendations = []
        
        # Sort mistake types by frequency
        sorted_patterns = sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True)
        
        for mistake_type, mistake_list in sorted_patterns:
            if not mistake_list:
                continue
            
            desc = FeedbackGenerator.MISTAKE_DESCRIPTIONS.get(mistake_type, {
                'title': 'Strategy Adjustment',
                'issue': 'Strategic deviations detected',
                'impact': 'Could be costing you money',
                'advice': ['Review your decision-making process']
            })
            
            count = len(mistake_list)
            frequency = f"{count} times ({(count/num_hands)*100:.0f}% of hands)"
            
            # Generate examples
            examples = []
            for mistake in mistake_list[:3]:
                example = f"Hand #{mistake.hand_id}: {mistake.situation}"
                if mistake.hand_type != 'unknown':
                    example += f" with {mistake.hand_type}"
                example += f" from {mistake.position}"
                example += f" → You {mistake.player_action}, Optimal: {mistake.optimal_action}"
                examples.append(example)
            
            # Estimate impact (rough BB/100 estimate)
            impact_per_mistake = 0.5  # Rough estimate
            total_bb_impact = count * impact_per_mistake
            bb_per_100 = (total_bb_impact / num_hands) * 100
            
            estimated_impact = f"+{bb_per_100:.1f} BB/100 hands potential improvement"
            
            recommendation = {
                'title': desc['title'],
                'issue': desc['issue'],
                'frequency': frequency,
                'examples': examples,
                'impact': desc['impact'],
                'advice': desc['advice'],
                'estimated_impact': estimated_impact,
                'count': count
            }
            
            recommendations.append(recommendation)
        
        return recommendations


