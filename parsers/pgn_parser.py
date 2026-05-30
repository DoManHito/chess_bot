import re
from dataclasses import dataclass, field
from typing import List, Optional
import chess
import chess.pgn


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
    """Smart streaming parser that searches for only 6% of games with Stockfish evaluations."""
    
    def parse_file(self, file_path: str) -> List[ParsedGame]:
        games = []
        
        print(f"Starting filtering and parsing file: {file_path}")
        print("Searching for games with built-in Stockfish analysis (the 6%)...")
        
        with open(file_path, 'r', encoding='utf-8') as pgn_file:
            checked_games = 0
            saved_games = 0
            
            while True:
                # Read one game from file (uses only a few KB of memory)
                lichess_game = chess.pgn.read_game(pgn_file)
                if lichess_game is None:
                    break  # Дошли до конца файла
                
                checked_games += 1
                
                # --- FILTERING TRICK ---
                # Peek at the first move of the game. If it doesn't contain '%eval',
                # then Lichess didn't analyze this game. Skip it entirely!
                first_move_node = lichess_game.next()
                if first_move_node is None or "%eval" not in first_move_node.comment:
                    continue  # Move to the next game
                
                parsed_game = ParsedGame()
                parsed_game.headers = dict(lichess_game.headers)
                parsed_game.result = parsed_game.headers.get("Result", "*")
                
                board = lichess_game.board()
                node = lichess_game
                turn_num = 1
                
                while node.variations:
                    next_node = node.variation(0)
                    move = next_node.move
                    
                    parsed_move = ParsedMove(
                        move=move.uci(),
                        san=board.san(move),
                        fen_before=board.fen(),
                        turn_num=turn_num,
                        turn_label="White" if board.turn == chess.WHITE else "Black"
                    )
                    
                    board.push(move)
                    parsed_move.fen_after = board.fen()
                    
                    # Extract evaluation [%eval 2.35] or [%eval #-4] (mate)
                    comment = next_node.comment
                    eval_match = re.search(r'%eval\s+([+-]?#?\d+\.?\d*)', comment)
                    
                    if eval_match:
                        eval_str = eval_match.group(1)
                        if '#' in eval_str:
                            # If it's mate (e.g., #4 or #-2), convert to a large evaluation
                            parsed_move.evaluation = 99.0 if '-' not in eval_str else -99.0
                        else:
                            parsed_move.evaluation = float(eval_str)
                    else:
                        parsed_move.evaluation = 0.0
                    
                    parsed_game.moves.append(parsed_move)
                    
                    node = next_node
                    if board.turn == chess.WHITE:
                        turn_num += 1
                
                games.append(parsed_game)
                saved_games += 1
                
                if saved_games % 100 == 0:
                    print(f"Scanned games: {checked_games} | Found and saved games with AI: {saved_games}")
                
                if saved_games >= 100000:
                    print(f"\nSuccess! Collected first {saved_games} games with Stockfish evaluations.")
                    break
                    
        return games