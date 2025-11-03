"""
Parse .phhs (Poker Hand History) files into structured Python objects
"""

import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class Hand:
    """Represents a single poker hand"""
    hand_id: int
    variant: str
    antes: List[float]
    blinds: List[float]
    min_bet: float
    starting_stacks: List[float]
    actions: List[str]
    venue: str
    time: str
    date: Dict[str, int]
    hand_number: int
    seats: List[int]
    table: str
    players: List[str]
    winnings: List[float]
    currency_symbol: str = '$'
    
    def get_player_index(self, player_name: str) -> Optional[int]:
        """Get the index of a player by name"""
        try:
            return self.players.index(player_name)
        except ValueError:
            return None
    
    def get_player_position(self, player_name: str) -> Optional[str]:
        """Get the position of a player (BTN, CO, UTG, etc.)"""
        player_idx = self.get_player_index(player_name)
        if player_idx is None:
            return None
        
        num_players = len(self.players)
        seat = self.seats[player_idx]
        
        # Button is the last to act pre-flop (after blinds)
        # Find button relative to blinds
        button_position = num_players - 1
        
        # Calculate position relative to button
        positions = ['BB', 'SB', 'BTN', 'CO', 'HJ', 'MP', 'MP', 'UTG', 'UTG']
        
        if num_players == 2:
            return 'BTN' if player_idx == 0 else 'BB'
        elif num_players <= len(positions):
            # Reverse positions for proper ordering
            position_map = positions[-num_players:]
            return position_map[player_idx]
        else:
            return f'P{player_idx + 1}'
    
    def get_player_actions(self, player_name: str) -> List[str]:
        """Get all actions taken by a specific player"""
        player_idx = self.get_player_index(player_name)
        if player_idx is None:
            return []
        
        player_actions = []
        player_marker = f'p{player_idx + 1}'
        
        for action in self.actions:
            if action.startswith(player_marker + ' '):
                player_actions.append(action)
        
        return player_actions


class HandHistoryParser:
    """Parser for .phhs files"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.hands: List[Hand] = []
    
    def parse(self) -> List[Hand]:
        """Parse the entire file and return list of hands"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by hand markers [1], [2], etc.
        hand_pattern = r'^\[(\d+)\]'
        parts = re.split(hand_pattern, content, flags=re.MULTILINE)
        
        # Skip first empty part
        parts = parts[1:]
        
        # Process in pairs (hand_id, hand_content)
        for i in range(0, len(parts) - 1, 2):
            hand_id = int(parts[i])
            hand_content = parts[i + 1].strip()
            
            hand = self._parse_hand(hand_id, hand_content)
            if hand:
                self.hands.append(hand)
        
        return self.hands
    
    def _parse_hand(self, hand_id: int, content: str) -> Optional[Hand]:
        """Parse a single hand"""
        try:
            lines = content.split('\n')
            hand_data = {}
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Parse the value
                    hand_data[key] = self._parse_value(value)
            
            # Create Hand object
            return Hand(
                hand_id=hand_id,
                variant=hand_data.get('variant', 'NT'),
                antes=hand_data.get('antes', []),
                blinds=hand_data.get('blinds_or_straddles', []),
                min_bet=hand_data.get('min_bet', 0.0),
                starting_stacks=hand_data.get('starting_stacks', []),
                actions=hand_data.get('actions', []),
                venue=hand_data.get('venue', ''),
                time=hand_data.get('time', '00:00:00'),
                date={
                    'day': hand_data.get('day', 1),
                    'month': hand_data.get('month', 1),
                    'year': hand_data.get('year', 2009)
                },
                hand_number=hand_data.get('hand', 0),
                seats=hand_data.get('seats', []),
                table=hand_data.get('table', ''),
                players=hand_data.get('players', []),
                winnings=hand_data.get('winnings', []),
                currency_symbol=hand_data.get('currency_symbol', '$')
            )
        except Exception as e:
            print(f"Error parsing hand {hand_id}: {e}")
            return None
    
    def _parse_value(self, value: str) -> Any:
        """Parse a value from the hand history"""
        # Remove quotes if present
        value = value.strip()
        if value.startswith("'") and value.endswith("'"):
            return value[1:-1]
        
        # Parse lists
        if value.startswith('[') and value.endswith(']'):
            # Extract list content
            list_content = value[1:-1]
            if not list_content.strip():
                return []
            
            # Split by comma, handling quoted strings
            items = []
            current = ''
            in_quotes = False
            
            for char in list_content:
                if char == "'" or char == '"':
                    in_quotes = not in_quotes
                    current += char
                elif char == ',' and not in_quotes:
                    items.append(current.strip())
                    current = ''
                else:
                    current += char
            
            if current.strip():
                items.append(current.strip())
            
            # Parse each item
            parsed_items = []
            for item in items:
                item = item.strip()
                if item.startswith("'") and item.endswith("'"):
                    parsed_items.append(item[1:-1])
                elif item.startswith('"') and item.endswith('"'):
                    parsed_items.append(item[1:-1])
                else:
                    try:
                        parsed_items.append(float(item))
                    except:
                        parsed_items.append(item)
            
            return parsed_items
        
        # Parse numbers
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except:
            return value
    
    def get_player_hands(self, player_name: str) -> List[Hand]:
        """Get all hands where a specific player participated"""
        return [hand for hand in self.hands if player_name in hand.players]


def parse_hand_history(file_path: str, player_name: Optional[str] = None) -> List[Hand]:
    """
    Convenience function to parse a hand history file
    
    Args:
        file_path: Path to .phhs file
        player_name: Optional player name to filter hands
    
    Returns:
        List of Hand objects
    """
    parser = HandHistoryParser(file_path)
    hands = parser.parse()
    
    if player_name:
        hands = [h for h in hands if player_name in h.players]
    
    return hands


