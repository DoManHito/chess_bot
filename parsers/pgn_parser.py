import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ParsedMove:
    """Represents a parsed chess move with all associated metadata."""
    move: str
    san: str
    fen_before: Optional[str] = None
    fen_after: Optional[str] = None
    turn_num: Optional[int] = None
    turn_label: Optional[str] = None
    evaluation: Optional[float] = None


@dataclass
class ParsedGame:
    """Represents a parsed chess game with headers and moves."""
    headers: dict = field(default_factory=dict)
    moves: List[ParsedMove] = field(default_factory=list)
    evaluation: Optional[float] = None
    result: Optional[str] = None


class PGNParser:
    """Parser for PGN (Portable Game Notation) chess files."""

    # Regex patterns
    HEADER_PATTERN = re.compile(r'\[([A-Za-z0-9_\-]+)\s+"(.+)"\]')
    MOVE_NUMBER_PATTERN = re.compile(r'(\d+\. (?:\d+\. )?)')
    EVAL_PATTERN = re.compile(r'\{(\+?[\d.]+)\}')
    EVAL_PATTERN_NO_BRACKETS = re.compile(r'(\+?[\d.]+)')
    SAN_PATTERN = re.compile(r'[a-h][1-8][a-h][1-8]|[a-h][1-8][=nbrq]|[a-h][=nbrq]|[Oo][o][o]|0-0-0|0-0|K[+-]?[a-h][1-8]?|k[+-]?[a-h][1-8]?|[NBRQKbnrqk]\d+[=nbrq]?|[NBRQKbnrqk][a-h][1-8]?|[=nbrq]')

    def __init__(self):
        """Initialize the PGN parser."""
        pass

    def _extract_headers(self, content: str) -> dict:
        """Extract PGN headers from content."""
        headers = {}
        for match in self.HEADER_PATTERN.finditer(content):
            key = match.group(1)
            value = match.group(2)
            headers[key] = value
        return headers

    def _extract_evaluation(self, move_text: str) -> Optional[float]:
        """Extract evaluation score from move annotation."""
        # First try with brackets {+0.55}
        match = self.EVAL_PATTERN.search(move_text)
        if match:
            eval_str = match.group(1)
            return float(eval_str)
        
        # Try without brackets +0.55
        match = self.EVAL_PATTERN_NO_BRACKETS.search(move_text)
        if match:
            eval_str = match.group(1)
            return float(eval_str)
        
        return None

    def _parse_moves(self, content: str) -> List[ParsedMove]:
        """Parse moves from PGN content."""
        moves = []
        lines = content.strip().split('\n')
        
        current_turn_num = 1
        current_turn_label = 'w'
        fen_before = None
        fen_after = None
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('('):
                continue
            
            # Check for move number
            move_match = self.MOVE_NUMBER_PATTERN.search(line)
            
            if move_match:
                # Extract turn number and label
                turn_num_str = move_match.group(1)
                current_turn_num = int(turn_num_str.split('.')[0])
                
                # Determine turn label based on turn number
                # White moves on odd turns, black on even
                current_turn_label = 'w' if current_turn_num % 2 == 1 else 'b'
                
                # Extract evaluation from this line
                evaluation = self._extract_evaluation(line)
                
                # Extract SAN notation (the actual move)
                san_match = self.SAN_PATTERN.search(line)
                san = san_match.group(0) if san_match else line.strip()
                
                # Remove evaluation from SAN if present
                san = self.EVAL_PATTERN.sub('', san).strip()
                
                move = ParsedMove(
                    move=line.strip(),
                    san=san,
                    turn_num=current_turn_num,
                    turn_label=current_turn_label,
                    evaluation=evaluation
                )
                moves.append(move)
                
                # Try to extract FEN before/after if available
                # This would require a FEN parser, which is optional
                # For now, we leave these as None
                
            else:
                # No move number, might be a continuation or annotation
                # Check if it's a continuation (no move number)
                if not line.startswith('{') and not line.startswith('['):
                    san_match = self.SAN_PATTERN.search(line)
                    if san_match:
                        san = san_match.group(0)
                        evaluation = self._extract_evaluation(line)
                        move = ParsedMove(
                            move=line.strip(),
                            san=san,
                            turn_label=current_turn_label,
                            evaluation=evaluation
                        )
                        moves.append(move)
        
        return moves

    def parse_file(self, file_path: str) -> List[ParsedGame]:
        """Parse PGN file containing one or more games."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self.parse_string(content)

    def parse_string(self, pgn_content: str) -> List[ParsedGame]:
        """Parse PGN content from string."""
        games = []
        
        # Split content into individual games
        # Games are separated by empty lines or new game markers
        game_blocks = re.split(r'\n\s*\n\s*\[Event', pgn_content)
        
        # First block might be empty or contain headers from previous game
        if game_blocks and game_blocks[0].strip():
            # Check if first block has game content
            if '[' not in game_blocks[0]:
                game_blocks = game_blocks[1:]
        
        for block in game_blocks:
            if not block.strip():
                continue
            
            game = ParsedGame()
            
            # Extract headers
            headers = self._extract_headers(block)
            game.headers = headers
            
            # Extract result if present
            if 'Result' in headers:
                game.result = headers['Result']
            
            # Extract FEN if present
            if 'FEN' in headers:
                game.evaluation = self._extract_evaluation(headers['FEN'])
            
            # Parse moves
            moves = self._parse_moves(block)
            game.moves = moves
            
            games.append(game)
        
        return games
