"""
Evaluate poker hand strength
"""

from typing import List, Tuple, Optional


class HandStrengthEvaluator:
    """Evaluate the strength of poker hands"""
    
    # Hand rankings (simplified Chen formula)
    HAND_SCORES = {
        # Premium pairs
        'AA': 10, 'KK': 10, 'QQ': 9,
        # High pairs
        'JJ': 8, 'TT': 7, '99': 7,
        # Medium pairs
        '88': 6, '77': 6, '66': 5, '55': 5,
        # Small pairs
        '44': 4, '33': 4, '22': 3,
        # Broadway
        'AKs': 9, 'AKo': 8, 'AQs': 8, 'AQo': 7,
        'AJs': 8, 'AJo': 6, 'ATs': 7, 'ATo': 6,
        'KQs': 7, 'KQo': 6, 'KJs': 7, 'KJo': 5,
        'KTs': 6, 'KTo': 5, 'QJs': 6, 'QJo': 5,
        'QTs': 6, 'QTo': 4, 'JTs': 6, 'JTo': 4,
        # Suited connectors
        'T9s': 5, '98s': 5, '87s': 4, '76s': 4,
        '65s': 4, '54s': 3,
    }
    
    @staticmethod
    def parse_cards(cards_str: str) -> Optional[List[str]]:
        """Parse card string like '????', 'AsKh', etc."""
        if '?' in cards_str:
            return None
        
        # Parse individual cards (format: RankSuit, e.g., 'As', 'Kh')
        cards = []
        i = 0
        while i < len(cards_str):
            if i + 1 < len(cards_str):
                rank = cards_str[i]
                suit = cards_str[i + 1]
                cards.append(rank + suit)
                i += 2
            else:
                i += 1
        
        return cards if len(cards) == 2 else None
    
    @staticmethod
    def get_hand_type(cards: List[str]) -> str:
        """Get hand type string (e.g., 'AKs', 'QQ', 'T9o')"""
        if len(cards) != 2:
            return 'unknown'
        
        rank1 = cards[0][0]
        suit1 = cards[0][1] if len(cards[0]) > 1 else ''
        rank2 = cards[1][0]
        suit2 = cards[1][1] if len(cards[1]) > 1 else ''
        
        # Normalize rank order (higher first)
        rank_order = 'AKQJT98765432'
        if rank_order.index(rank1) > rank_order.index(rank2):
            rank1, rank2 = rank2, rank1
            suit1, suit2 = suit2, suit1
        
        # Check if pair
        if rank1 == rank2:
            return f'{rank1}{rank2}'
        
        # Check if suited
        suited = 's' if suit1 == suit2 else 'o'
        
        return f'{rank1}{rank2}{suited}'
    
    @classmethod
    def evaluate_preflop_strength(cls, cards: List[str]) -> Tuple[int, str]:
        """
        Evaluate preflop hand strength
        
        Returns:
            (score, hand_type) where score is 1-10
        """
        hand_type = cls.get_hand_type(cards)
        score = cls.HAND_SCORES.get(hand_type, 2)  # Default low score
        
        return score, hand_type
    
    @staticmethod
    def get_position_strength(position: str) -> int:
        """Get relative strength of position (higher is better)"""
        position_rank = {
            'BTN': 5,  # Button - best
            'CO': 4,   # Cut-off
            'HJ': 3,   # Hijack
            'MP': 2,   # Middle
            'UTG': 1,  # Under the gun - worst
            'SB': 1,   # Small blind
            'BB': 1,   # Big blind
        }
        return position_rank.get(position, 2)

