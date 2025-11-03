"""
Simplified GTO (Game Theory Optimal) strategy baseline
"""

from typing import Dict, Tuple, Optional, List
from .hand_strength import HandStrengthEvaluator


class StrategyBaseline:
    """Simplified GTO strategy for poker"""
    
    # Preflop opening ranges by position (minimum hand score)
    OPENING_RANGES = {
        'UTG': 7,    # Tight: QQ+, AK (top 8%)
        'MP': 6,     # Medium: 99+, AJ+ (top 15%)
        'HJ': 5,     # Loose-medium: 77+, AT+ (top 20%)
        'CO': 4,     # Loose: 66+, A9+, KJ+ (top 25%)
        'BTN': 3,    # Very loose: Any pair, any ace, suited connectors (top 40%)
        'SB': 5,     # Similar to MP
        'BB': 6,     # Defend wide but not super loose
    }
    
    # Facing a raise: calling ranges
    CALLING_RANGES = {
        'IP': 5,   # In position: call with decent hands
        'OOP': 6,  # Out of position: need stronger hands
    }
    
    # 3-bet ranges
    THREE_BET_RANGE = 8  # Strong hands only
    
    # C-bet frequencies by situation
    CBET_FREQUENCY = {
        'heads_up': 0.65,   # 65% of time heads up
        'multiway': 0.45,   # 45% of time multiway
    }
    
    @classmethod
    def get_optimal_preflop_action(
        cls,
        hand_score: int,
        position: str,
        facing_action: str = 'none',
        num_opponents: int = 1
    ) -> Tuple[str, float, str]:
        """
        Get optimal preflop action
        
        Args:
            hand_score: Hand strength score (1-10)
            position: Player position
            facing_action: 'none', 'raise', '3bet'
            num_opponents: Number of active opponents
        
        Returns:
            (action, probability, reasoning)
            action: 'fold', 'call', 'raise'
        """
        
        # Facing no action - should we open?
        if facing_action == 'none':
            min_score = cls.OPENING_RANGES.get(position, 6)
            if hand_score >= min_score:
                return ('raise', 0.90, f'Open raise from {position} with strong enough hand')
            else:
                return ('fold', 0.90, f'Hand too weak to open from {position}')
        
        # Facing a raise - should we call/3-bet/fold?
        elif facing_action == 'raise':
            # In position or out of position?
            in_position = position in ['BTN', 'CO']
            min_call_score = cls.CALLING_RANGES['IP'] if in_position else cls.CALLING_RANGES['OOP']
            
            # Premium hands: 3-bet
            if hand_score >= cls.THREE_BET_RANGE:
                return ('raise', 0.75, '3-bet with premium hand')
            
            # Good hands: call
            elif hand_score >= min_call_score:
                return ('call', 0.70, f'Call with decent hand {"in position" if in_position else "out of position"}')
            
            # Weak hands: fold
            else:
                return ('fold', 0.85, 'Hand too weak to continue vs raise')
        
        # Facing a 3-bet
        elif facing_action == '3bet':
            # Only continue with very strong hands
            if hand_score >= 9:
                return ('call', 0.60, 'Call/4-bet with premium hand vs 3-bet')
            else:
                return ('fold', 0.80, 'Fold to 3-bet without premium hand')
        
        return ('fold', 0.50, 'Default action')
    
    @classmethod
    def get_optimal_postflop_action(
        cls,
        raised_preflop: bool,
        is_heads_up: bool,
        position: str
    ) -> Tuple[str, float, str]:
        """
        Get optimal postflop c-bet action
        
        Returns:
            (action, probability, reasoning)
        """
        
        if not raised_preflop:
            return ('check', 0.70, 'Did not raise preflop')
        
        # C-bet frequency
        cbet_freq = cls.CBET_FREQUENCY['heads_up'] if is_heads_up else cls.CBET_FREQUENCY['multiway']
        
        # In position is better
        if position in ['BTN', 'CO']:
            cbet_freq += 0.10
        
        if cbet_freq >= 0.55:
            return ('bet', cbet_freq, f'C-bet {"heads up" if is_heads_up else "multiway"}')
        else:
            return ('check', 1 - cbet_freq, 'Check multiway')
    
    @classmethod
    def evaluate_player_action(
        cls,
        player_action: str,
        optimal_action: str,
        optimal_probability: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluate if player's action matches baseline
        
        Returns:
            (is_mistake, mistake_type)
        """
        
        # Normalize actions
        action_map = {
            'f': 'fold',
            'cc': 'call',
            'cbr': 'raise',
            'cr': 'raise',
        }
        
        player_act = action_map.get(player_action, player_action)
        
        # If actions match, no mistake
        if player_act == optimal_action:
            return (False, None)
        
        # Identify mistake type
        if player_act == 'fold' and optimal_action in ['call', 'raise']:
            if optimal_probability > 0.60:
                return (True, 'over_folding')
        
        elif player_act == 'call' and optimal_action == 'fold':
            if optimal_probability > 0.70:
                return (True, 'over_calling')
        
        elif player_act == 'call' and optimal_action == 'raise':
            if optimal_probability > 0.60:
                return (True, 'under_aggressive')
        
        elif player_act == 'fold' and optimal_action == 'call':
            if optimal_probability > 0.60:
                return (True, 'over_folding')
        
        # If low probability optimal action, not a big mistake
        if optimal_probability < 0.55:
            return (False, None)
        
        return (True, 'strategy_deviation')

