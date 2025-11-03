"""
Calculate poker statistics from hand histories
"""

from typing import List, Dict
from collections import defaultdict
from .parser import Hand


class PokerStatistics:
    """Calculate various poker statistics for a player"""
    
    def __init__(self, hands: List[Hand], player_name: str):
        self.hands = hands
        self.player_name = player_name
        self.stats = {}
        
    def calculate_all(self) -> Dict:
        """Calculate all statistics"""
        self.stats = {
            'hands_played': len(self.hands),
            'vpip': self._calculate_vpip(),
            'pfr': self._calculate_pfr(),
            'three_bet_pct': self._calculate_three_bet(),
            'fold_to_three_bet_pct': self._calculate_fold_to_three_bet(),
            'cbet_pct': self._calculate_cbet(),
            'aggression_factor': self._calculate_aggression(),
            'position_stats': self._calculate_position_stats(),
            'win_rate': self._calculate_win_rate()
        }
        return self.stats
    
    def _get_player_preflop_action(self, hand: Hand) -> str:
        """Get player's first voluntary preflop action"""
        player_idx = hand.get_player_index(self.player_name)
        if player_idx is None:
            return 'fold'
        
        player_marker = f'p{player_idx + 1}'
        
        # Find first action (skip dealing actions)
        for action in hand.actions:
            if action.startswith('d '):  # Skip dealing
                continue
            if action.startswith(player_marker + ' '):
                # Extract action type
                parts = action.split()
                if len(parts) >= 2:
                    action_type = parts[1]
                    return action_type
        
        return 'fold'
    
    def _calculate_vpip(self) -> float:
        """VPIP: Voluntarily Put $ In Pot %"""
        if not self.hands:
            return 0.0
        
        voluntary_hands = 0
        for hand in self.hands:
            action = self._get_player_preflop_action(hand)
            # Count call, raise, or bet as voluntary
            if action in ['cc', 'cbr', 'cr']:  # call, raise
                voluntary_hands += 1
        
        return (voluntary_hands / len(self.hands)) * 100
    
    def _calculate_pfr(self) -> float:
        """PFR: Pre-Flop Raise %"""
        if not self.hands:
            return 0.0
        
        raise_hands = 0
        for hand in self.hands:
            action = self._get_player_preflop_action(hand)
            if action in ['cbr', 'cr']:  # raise
                raise_hands += 1
        
        return (raise_hands / len(self.hands)) * 100
    
    def _calculate_three_bet(self) -> float:
        """3-bet percentage"""
        opportunities = 0
        three_bets = 0
        
        for hand in self.hands:
            player_idx = hand.get_player_index(self.player_name)
            if player_idx is None:
                continue
            
            player_marker = f'p{player_idx + 1}'
            
            # Check if there was a raise before player acted
            saw_raise = False
            player_three_bet = False
            
            for action in hand.actions:
                if action.startswith('d '):
                    continue
                    
                # Check if someone raised
                if 'cbr' in action or 'cr' in action:
                    if not action.startswith(player_marker):
                        saw_raise = True
                
                # Check if player 3-bet
                if action.startswith(player_marker) and saw_raise:
                    if 'cbr' in action or 'cr' in action:
                        player_three_bet = True
                    break
            
            if saw_raise:
                opportunities += 1
                if player_three_bet:
                    three_bets += 1
        
        return (three_bets / opportunities * 100) if opportunities > 0 else 0.0
    
    def _calculate_fold_to_three_bet(self) -> float:
        """Fold to 3-bet percentage"""
        opportunities = 0
        folds = 0
        
        for hand in self.hands:
            player_idx = hand.get_player_index(self.player_name)
            if player_idx is None:
                continue
            
            player_marker = f'p{player_idx + 1}'
            
            # Check if player raised, then faced a 3-bet
            player_raised = False
            faced_three_bet = False
            player_folded = False
            
            for action in hand.actions:
                if action.startswith('d '):
                    continue
                
                if action.startswith(player_marker) and ('cbr' in action or 'cr' in action):
                    player_raised = True
                
                if player_raised and not action.startswith(player_marker):
                    if 'cbr' in action or 'cr' in action:
                        faced_three_bet = True
                
                if faced_three_bet and action.startswith(player_marker):
                    if 'f' == action.split()[1]:
                        player_folded = True
                    break
            
            if faced_three_bet:
                opportunities += 1
                if player_folded:
                    folds += 1
        
        return (folds / opportunities * 100) if opportunities > 0 else 0.0
    
    def _calculate_cbet(self) -> float:
        """Continuation bet percentage"""
        opportunities = 0
        cbets = 0
        
        for hand in self.hands:
            player_idx = hand.get_player_index(self.player_name)
            if player_idx is None:
                continue
            
            player_marker = f'p{player_idx + 1}'
            
            # Check if player raised preflop
            raised_preflop = False
            saw_flop = False
            cbet_made = False
            
            for action in hand.actions:
                # Check for preflop raise
                if not saw_flop and action.startswith(player_marker):
                    if 'cbr' in action or 'cr' in action:
                        raised_preflop = True
                
                # Check for flop (board dealt)
                if action.startswith('d db'):
                    saw_flop = True
                
                # Check for c-bet on flop
                if saw_flop and raised_preflop and action.startswith(player_marker):
                    if 'cbr' in action or 'cr' in action:
                        cbet_made = True
                    break
            
            if raised_preflop and saw_flop:
                opportunities += 1
                if cbet_made:
                    cbets += 1
        
        return (cbets / opportunities * 100) if opportunities > 0 else 0.0
    
    def _calculate_aggression(self) -> float:
        """Aggression factor: (bets + raises) / calls"""
        bets_raises = 0
        calls = 0
        
        for hand in self.hands:
            player_idx = hand.get_player_index(self.player_name)
            if player_idx is None:
                continue
            
            player_marker = f'p{player_idx + 1}'
            
            for action in hand.actions:
                if action.startswith(player_marker):
                    parts = action.split()
                    if len(parts) >= 2:
                        action_type = parts[1]
                        if action_type in ['cbr', 'cr']:
                            bets_raises += 1
                        elif action_type == 'cc':
                            calls += 1
        
        return (bets_raises / calls) if calls > 0 else 0.0
    
    def _calculate_position_stats(self) -> Dict:
        """Calculate stats by position"""
        position_data = defaultdict(lambda: {'count': 0, 'vpip': 0, 'pfr': 0})
        
        for hand in self.hands:
            position = hand.get_player_position(self.player_name)
            if position:
                position_data[position]['count'] += 1
                
                action = self._get_player_preflop_action(hand)
                if action in ['cc', 'cbr', 'cr']:
                    position_data[position]['vpip'] += 1
                if action in ['cbr', 'cr']:
                    position_data[position]['pfr'] += 1
        
        # Calculate percentages
        result = {}
        for pos, data in position_data.items():
            if data['count'] > 0:
                result[pos] = {
                    'hands': data['count'],
                    'vpip': (data['vpip'] / data['count']) * 100,
                    'pfr': (data['pfr'] / data['count']) * 100
                }
        
        return result
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate in BB/hand"""
        if not self.hands:
            return 0.0
        
        total_winnings = 0
        for hand in self.hands:
            player_idx = hand.get_player_index(self.player_name)
            if player_idx is not None and player_idx < len(hand.winnings):
                total_winnings += hand.winnings[player_idx]
        
        # Convert to BB (big blinds)
        avg_bb = 0.50  # Assuming $0.50 BB from the data
        bb_won = total_winnings / avg_bb
        
        return bb_won / len(self.hands)


