"""
Analyze player actions against strategy baseline
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .parser import Hand
from .hand_strength import HandStrengthEvaluator
from .strategy import StrategyBaseline


@dataclass
class Mistake:
    """Represents a strategic mistake"""
    hand_id: int
    hand_type: str
    position: str
    situation: str
    player_action: str
    optimal_action: str
    optimal_probability: float
    mistake_type: str
    reasoning: str
    severity: str = 'medium'


class PlayerAnalyzer:
    """Analyze a player's strategy against baseline"""
    
    def __init__(self, hands: List[Hand], player_name: str):
        self.hands = hands
        self.player_name = player_name
        self.mistakes: List[Mistake] = []
        self.evaluator = HandStrengthEvaluator()
        self.strategy = StrategyBaseline()
    
    def analyze_all_hands(self) -> Tuple[List[Mistake], Dict]:
        """
        Analyze all hands and identify mistakes
        
        Returns:
            (mistakes, summary_stats)
        """
        self.mistakes = []
        
        for hand in self.hands:
            self._analyze_hand(hand)
        
        # Generate summary
        summary = self._generate_summary()
        
        return self.mistakes, summary
    
    def _analyze_hand(self, hand: Hand):
        """Analyze a single hand for mistakes"""
        player_idx = hand.get_player_index(self.player_name)
        if player_idx is None:
            return
        
        player_marker = f'p{player_idx + 1}'
        position = hand.get_player_position(self.player_name)
        
        # Extract player's hole cards if visible
        hole_cards = None
        hand_score = 5  # Default medium strength
        hand_type = 'unknown'
        
        for action in hand.actions:
            if action.startswith(f'd dh {player_marker}'):
                cards_str = action.split()[-1]
                cards = self.evaluator.parse_cards(cards_str)
                if cards:
                    hand_score, hand_type = self.evaluator.evaluate_preflop_strength(cards)
                    hole_cards = cards
                break
        
        # Analyze preflop action
        self._analyze_preflop_action(
            hand, player_marker, position, hand_score, hand_type
        )
    
    def _analyze_preflop_action(
        self,
        hand: Hand,
        player_marker: str,
        position: str,
        hand_score: int,
        hand_type: str
    ):
        """Analyze player's preflop action"""
        
        # Determine what action player faced
        facing_action = 'none'
        saw_raise_before = False
        saw_three_bet_before = False
        player_raised = False
        
        for action in hand.actions:
            if action.startswith('d '):
                continue
            
            # Check actions before player
            if not action.startswith(player_marker):
                if 'cbr' in action or 'cr' in action:
                    if player_raised:
                        saw_three_bet_before = True
                    else:
                        saw_raise_before = True
            
            # Player's action
            if action.startswith(player_marker):
                parts = action.split()
                if len(parts) >= 2:
                    player_action = parts[1]
                    
                    # Determine situation
                    if saw_three_bet_before:
                        facing_action = '3bet'
                    elif saw_raise_before:
                        facing_action = 'raise'
                    else:
                        facing_action = 'none'
                    
                    # Get optimal action
                    num_opponents = len(hand.players) - 1
                    optimal_action, optimal_prob, reasoning = \
                        self.strategy.get_optimal_preflop_action(
                            hand_score, position, facing_action, num_opponents
                        )
                    
                    # Check if mistake
                    is_mistake, mistake_type = self.strategy.evaluate_player_action(
                        player_action, optimal_action, optimal_prob
                    )
                    
                    if is_mistake:
                        # Determine severity
                        severity = 'high' if optimal_prob > 0.75 else 'medium'
                        
                        mistake = Mistake(
                            hand_id=hand.hand_id,
                            hand_type=hand_type if hand_type != 'unknown' else f'score_{hand_score}',
                            position=position or 'unknown',
                            situation=f'Preflop facing {facing_action}',
                            player_action=player_action,
                            optimal_action=optimal_action,
                            optimal_probability=optimal_prob,
                            mistake_type=mistake_type,
                            reasoning=reasoning,
                            severity=severity
                        )
                        self.mistakes.append(mistake)
                    
                    # Track if player raised (for 3-bet detection)
                    if player_action in ['cbr', 'cr']:
                        player_raised = True
                    
                    break
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics of mistakes"""
        if not self.mistakes:
            return {
                'total_mistakes': 0,
                'agreement_rate': 100.0,
                'mistake_breakdown': {}
            }
        
        # Count mistakes by type
        mistake_counts = defaultdict(int)
        for mistake in self.mistakes:
            mistake_counts[mistake.mistake_type] += 1
        
        # Agreement rate
        total_decisions = len(self.hands)
        mistakes = len(self.mistakes)
        agreement_rate = ((total_decisions - mistakes) / total_decisions) * 100
        
        return {
            'total_mistakes': mistakes,
            'total_hands': total_decisions,
            'agreement_rate': agreement_rate,
            'mistake_breakdown': dict(mistake_counts)
        }
    
    def get_patterns(self) -> Dict[str, List[Mistake]]:
        """Group mistakes by type for pattern analysis"""
        patterns = defaultdict(list)
        for mistake in self.mistakes:
            patterns[mistake.mistake_type].append(mistake)
        return dict(patterns)
    
    def get_position_mistakes(self) -> Dict[str, int]:
        """Count mistakes by position"""
        position_mistakes = defaultdict(int)
        for mistake in self.mistakes:
            position_mistakes[mistake.position] += 1
        return dict(position_mistakes)

